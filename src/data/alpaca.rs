//! Alpaca instruct rows. The prompt and the response are tokenized separately
//! and concatenated; the response ends with eos and is the only part labelled
//! unless `train_on_inputs`.

use super::text::{render, Message, Text};
use crate::config::Dataset;
use crate::{Example, Prompt, IGNORE_INDEX};
use anyhow::{Context, Result};
use serde_json::Value;

pub const SYSTEM: &str = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.";
pub const SYSTEM_NO_INPUT: &str = "Below is an instruction that describes a task. Write a response that appropriately completes the request.";

pub fn prompt_text(instruction: &str, input: &str) -> String {
    if input.is_empty() {
        format!("{SYSTEM_NO_INPUT}\n\n### Instruction:\n{instruction}\n\n### Response:\n")
    } else {
        format!(
            "{SYSTEM}\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
        )
    }
}

fn fields<'a>(ds: &Dataset, row: &'a Value) -> Result<(&'a str, &'a str, &'a str)> {
    let s = |k: &str| {
        row[k]
            .as_str()
            .with_context(|| format!("row has no `{k}` string"))
    };
    Ok((
        s(&ds.field_instruction)?,
        row[&ds.field_input].as_str().unwrap_or(""),
        s(&ds.field_output)?,
    ))
}

pub fn example(ds: &Dataset, text: &Text, train_on_inputs: bool, row: &Value) -> Result<Example> {
    let (instruction, input, output) = fields(ds, row)?;
    let p = text.encode(&prompt_text(instruction, input), true)?;
    let mut r = text.encode(output, true)?;
    if text.bos_id.is_some() && r.first() == text.bos_id.as_ref() {
        r.remove(0);
    }
    if r.last() != Some(&text.eos_id) {
        r.push(text.eos_id);
    }
    let mut labels: Vec<i64> = p
        .iter()
        .map(|&i| {
            if train_on_inputs {
                i as i64
            } else {
                IGNORE_INDEX
            }
        })
        .collect();
    labels.extend(r.iter().map(|&i| i as i64));
    let mut input_ids = p;
    input_ids.extend(r);
    Ok(Example { input_ids, labels })
}

/// The instruction (and input) as one user turn through the chat template;
/// the output is the answer.
pub fn prompt(ds: &Dataset, text: &Text, row: &Value) -> Result<Prompt> {
    let (instruction, input, output) = fields(ds, row)?;
    let content = if input.is_empty() {
        instruction.to_string()
    } else {
        format!("{instruction}\n\n{input}")
    };
    let s = render(
        text,
        &[Message {
            role: "user".into(),
            content,
        }],
        true,
    )?;
    Ok(Prompt {
        input_ids: text.encode(&s, false)?,
        text: s,
        answer: Some(output.to_string()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::DatasetKind;
    use crate::data::fixture;
    use serde_json::json;

    #[test]
    fn strings_are_byte_exact() {
        assert_eq!(
            prompt_text("Say hi", ""),
            "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nSay hi\n\n### Response:\n"
        );
        assert_eq!(
            prompt_text("Say hi", "to Bob"),
            "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\nSay hi\n\n### Input:\nto Bob\n\n### Response:\n"
        );
    }

    #[test]
    fn mask_and_eos() {
        let text = fixture::text();
        let ds = fixture::dataset(DatasetKind::Alpaca, "");
        let row = json!({ "instruction": "hi there", "input": "", "output": "fine thanks" });
        let n = text
            .encode(&prompt_text("hi there", ""), true)
            .unwrap()
            .len();
        let ex = example(&ds, &text, false, &row).unwrap();
        assert_eq!(ex.input_ids.len(), n + 3);
        assert!(ex.labels[..n].iter().all(|&l| l == IGNORE_INDEX));
        assert_eq!(
            &ex.labels[n..],
            &[
                ex.input_ids[n] as i64,
                ex.input_ids[n + 1] as i64,
                text.eos_id as i64
            ]
        );
        assert_eq!(ex.input_ids[n], text.tokenizer.token_to_id("fine").unwrap());
        let ex = example(&ds, &text, true, &row).unwrap();
        assert!(ex
            .labels
            .iter()
            .zip(&ex.input_ids)
            .all(|(&l, &i)| l == i as i64));
        assert!(example(&ds, &text, false, &json!({ "instruction": "x" })).is_err());
    }

    #[test]
    fn response_drops_leading_bos_and_keeps_one_eos() {
        let mut text = fixture::text();
        text.bos_id = text.tokenizer.token_to_id("<|endoftext|>");
        let ds = fixture::dataset(DatasetKind::Alpaca, "");
        let row = json!({ "instruction": "hi", "output": "<|endoftext|> fine <|im_end|>" });
        let ex = example(&ds, &text, false, &row).unwrap();
        let tail = &ex.input_ids[ex.input_ids.len() - 2..];
        assert_eq!(
            tail,
            &[text.tokenizer.token_to_id("fine").unwrap(), text.eos_id]
        );
        assert_eq!(
            ex.input_ids.iter().filter(|&&i| i == text.eos_id).count(),
            1
        );
    }

    #[test]
    fn prompt_goes_through_chat_template() {
        let text = fixture::text();
        let ds = fixture::dataset(DatasetKind::Alpaca, "");
        let p = prompt(
            &ds,
            &text,
            &json!({ "instruction": "hi there", "input": "how are you", "output": "fine" }),
        )
        .unwrap();
        assert_eq!(
            p.text,
            "<|im_start|>user\nhi there\n\nhow are you<|im_end|>\n<|im_start|>assistant\n"
        );
        assert_eq!(p.answer.as_deref(), Some("fine"));
        assert_eq!(p.input_ids.len(), 10);
    }
}
