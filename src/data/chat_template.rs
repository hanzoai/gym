//! Conversations rendered through the chat template. Labels cover the content
//! of trainable turns, found by rendering the conversation up to each turn
//! twice (once with the turn's content replaced by `[[dummy_message]]`) and
//! diffing the token ids from both ends, plus end-of-sequence tokens as
//! `train_on_eos` asks.

use super::text::{render, Message, Text};
use crate::config::{Dataset, MessageMap, TrainOnEos};
use crate::{Example, Prompt, IGNORE_INDEX};
use anyhow::{Context, Result};
use serde_json::Value;

const DUMMY: &str = "[[dummy_message]]";

/// Canonical role for a source role name.
pub fn role(r: &str) -> &str {
    match r {
        "human" => "user",
        "gpt" => "assistant",
        _ => r,
    }
}

/// The row's turns with fields mapped and roles canonical, with a system turn
/// from the row's `system` field in front when the first turn is not system.
pub fn messages(ds: &Dataset, row: &Value) -> Result<Vec<Message>> {
    let map = ds.message_property_mappings.clone().unwrap_or(MessageMap {
        role: "role".into(),
        content: "content".into(),
    });
    let list = row[&ds.field_messages]
        .as_array()
        .with_context(|| format!("row has no `{}` array", ds.field_messages))?;
    let mut out: Vec<Message> = list
        .iter()
        .map(|m| {
            let field = |k: &str| {
                m[k].as_str()
                    .map(String::from)
                    .with_context(|| format!("message has no `{k}` string"))
            };
            Ok(Message {
                role: role(&field(&map.role)?).to_string(),
                content: field(&map.content)?,
            })
        })
        .collect::<Result<_>>()?;
    if let Some(sys) = row["system"].as_str() {
        if out.first().is_none_or(|m| m.role != "system") {
            out.insert(
                0,
                Message {
                    role: "system".into(),
                    content: sys.into(),
                },
            );
        }
    }
    Ok(out)
}

pub fn example(ds: &Dataset, text: &Text, train_on_inputs: bool, row: &Value) -> Result<Example> {
    let turns = messages(ds, row)?;
    let ids = text.encode(&render(text, &turns, false)?, false)?;
    let mut labels = vec![IGNORE_INDEX; ids.len()];
    let roles: Vec<&str> = ds.roles_to_train.iter().map(|r| role(r)).collect();
    let eos = ds.train_on_eos;
    let mut last_eos = None;
    for i in 0..turns.len() {
        let train = train_on_inputs || roles.contains(&turns[i].role.as_str());
        if !train && eos != TrainOnEos::All {
            continue;
        }
        let Some((start, end)) = span(text, &turns, i)? else {
            continue;
        };
        let end = end.min(ids.len());
        if train {
            for j in start..end {
                labels[j] = ids[j] as i64;
            }
        }
        let Some(e) = ids[end..]
            .iter()
            .position(|&t| t == text.eos_id)
            .map(|p| p + end)
        else {
            continue;
        };
        if e - end > 3 {
            continue;
        }
        last_eos = Some(e);
        if eos == TrainOnEos::All || (eos == TrainOnEos::Turn && train) {
            labels[e] = ids[e] as i64;
        }
    }
    if let (TrainOnEos::Last, Some(e)) = (eos, last_eos) {
        labels[e] = ids[e] as i64;
    }
    Ok(Example {
        input_ids: ids,
        labels,
    })
}

/// Token span `[start, end)` of turn `i`'s content within the render of
/// `turns[..=i]`, or `None` when the template leaves no trace of it.
fn span(text: &Text, turns: &[Message], i: usize) -> Result<Option<(usize, usize)>> {
    let mut dummy = turns[..=i].to_vec();
    dummy[i].content = DUMMY.into();
    let d = text.encode(&render(text, &dummy, false)?, false)?;
    let f = text.encode(&render(text, &turns[..=i], false)?, false)?;
    if d.is_empty() || f.is_empty() {
        return Ok(None);
    }
    let n = d.len().min(f.len());
    let Some(start) = (0..n).find(|&k| d[k] != f[k]) else {
        return Ok(None);
    };
    let Some(end) = (0..n)
        .find(|&k| d[d.len() - 1 - k] != f[f.len() - 1 - k])
        .map(|k| f.len() - k)
    else {
        return Ok(None);
    };
    Ok((end > start).then_some((start, end)))
}

/// Everything before the last assistant turn, rendered with a generation
/// prompt; that turn's content is the answer.
pub fn prompt(ds: &Dataset, text: &Text, row: &Value) -> Result<Prompt> {
    let turns = messages(ds, row)?;
    let (prefix, answer) = match turns.iter().rposition(|m| m.role == "assistant") {
        Some(i) => (&turns[..i], Some(turns[i].content.clone())),
        None => (&turns[..], super::answer(row)),
    };
    let s = render(text, prefix, true)?;
    Ok(Prompt {
        input_ids: text.encode(&s, false)?,
        text: s,
        answer,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::DatasetKind;
    use crate::data::fixture;
    use serde_json::json;
    use std::collections::BTreeSet;

    fn row() -> Value {
        json!({ "messages": [
            { "role": "system", "content": "be good" },
            { "role": "user", "content": "hi there" },
            { "role": "assistant", "content": "fine thanks" },
            { "role": "user", "content": "how are you" },
            { "role": "assistant", "content": "you too" },
        ] })
    }

    fn trained(ex: &Example) -> BTreeSet<usize> {
        (0..ex.labels.len())
            .filter(|&i| ex.labels[i] != IGNORE_INDEX)
            .collect()
    }

    fn set(v: &[usize]) -> BTreeSet<usize> {
        v.iter().copied().collect()
    }

    /// Tokens per turn: `[<|im_start|>, role, words..., <|im_end|>]`, so the
    /// five turns occupy 0..5, 5..10, 10..15, 15..21, 21..26.
    #[test]
    fn labels_per_train_on_eos() {
        let text = fixture::text();
        let mut ds = fixture::dataset(DatasetKind::ChatTemplate, "");
        let ex = example(&ds, &text, false, &row()).unwrap();
        assert_eq!(ex.input_ids.len(), 26);
        assert_eq!(ex.input_ids[14], text.eos_id);
        assert_eq!(ex.input_ids[25], text.eos_id);
        assert_eq!(trained(&ex), set(&[12, 13, 14, 23, 24, 25]));
        for i in trained(&ex) {
            assert_eq!(ex.labels[i], ex.input_ids[i] as i64);
        }

        ds.train_on_eos = TrainOnEos::All;
        assert_eq!(
            trained(&example(&ds, &text, false, &row()).unwrap()),
            set(&[4, 9, 12, 13, 14, 20, 23, 24, 25])
        );
        ds.train_on_eos = TrainOnEos::Last;
        assert_eq!(
            trained(&example(&ds, &text, false, &row()).unwrap()),
            set(&[12, 13, 23, 24, 25])
        );
        ds.train_on_eos = TrainOnEos::None;
        assert_eq!(
            trained(&example(&ds, &text, false, &row()).unwrap()),
            set(&[12, 13, 23, 24])
        );

        ds.train_on_eos = TrainOnEos::Turn;
        let all = example(&ds, &text, true, &row()).unwrap();
        assert_eq!(
            trained(&all),
            set(&[2, 3, 4, 7, 8, 9, 12, 13, 14, 17, 18, 19, 20, 23, 24, 25])
        );

        ds.roles_to_train = vec!["gpt".into(), "human".into()];
        assert_eq!(
            trained(&example(&ds, &text, false, &row()).unwrap()),
            set(&[7, 8, 9, 12, 13, 14, 17, 18, 19, 20, 23, 24, 25])
        );
    }

    #[test]
    fn sharegpt_mapping_and_system_field() {
        let text = fixture::text();
        let mut ds = fixture::dataset(DatasetKind::ChatTemplate, "");
        ds.field_messages = "conversations".into();
        ds.message_property_mappings = Some(MessageMap {
            role: "from".into(),
            content: "value".into(),
        });
        let r = json!({ "system": "be good", "conversations": [
            { "from": "human", "value": "hi there" },
            { "from": "gpt", "value": "fine thanks" },
        ] });
        let m = messages(&ds, &r).unwrap();
        assert_eq!(
            m.iter().map(|m| m.role.as_str()).collect::<Vec<_>>(),
            ["system", "user", "assistant"]
        );
        assert_eq!(m[0].content, "be good");
        assert_eq!(m[2].content, "fine thanks");
        assert_eq!(
            trained(&example(&ds, &text, false, &r).unwrap()),
            set(&[12, 13, 14])
        );
        assert!(messages(&ds, &json!({ "conversations": [{ "from": "human" }] })).is_err());
        assert!(messages(&ds, &json!({ "text": "x" })).is_err());
    }

    #[test]
    fn prompt_splits_final_assistant_turn() {
        let text = fixture::text();
        let ds = fixture::dataset(DatasetKind::ChatTemplate, "");
        let p = prompt(&ds, &text, &row()).unwrap();
        assert!(p
            .text
            .ends_with("how are you<|im_end|>\n<|im_start|>assistant\n"));
        assert_eq!(p.answer.as_deref(), Some("you too"));
        assert_eq!(p.input_ids.len(), 23);
        let r =
            json!({ "messages": [{ "role": "user", "content": "hi there" }], "answer": "fine" });
        let p = prompt(&ds, &text, &r).unwrap();
        assert_eq!(p.answer.as_deref(), Some("fine"));
        assert!(p.text.ends_with("<|im_start|>assistant\n"));
    }
}
