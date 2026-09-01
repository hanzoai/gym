//! Raw text rows: every token is trained and eos is appended once.

use super::text::Text;
use crate::config::Dataset;
use crate::{Example, Prompt};
use anyhow::{Context, Result};
use serde_json::Value;

fn field<'a>(ds: &Dataset, row: &'a Value) -> Result<&'a str> {
    row[&ds.field]
        .as_str()
        .with_context(|| format!("row has no `{}` string", ds.field))
}

pub fn example(ds: &Dataset, text: &Text, row: &Value) -> Result<Example> {
    let mut ids = text.encode(field(ds, row)?, true)?;
    if ids.last() != Some(&text.eos_id) {
        ids.push(text.eos_id);
    }
    let labels = ids.iter().map(|&i| i as i64).collect();
    Ok(Example {
        input_ids: ids,
        labels,
    })
}

/// The text itself is the prompt; `answer` or `solution` is the answer.
pub fn prompt(ds: &Dataset, text: &Text, row: &Value) -> Result<Prompt> {
    let s = field(ds, row)?.to_string();
    Ok(Prompt {
        input_ids: text.encode(&s, true)?,
        text: s,
        answer: super::answer(row),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::DatasetKind;
    use crate::data::fixture;
    use serde_json::json;

    #[test]
    fn all_tokens_trained_with_one_eos() {
        let text = fixture::text();
        let mut ds = fixture::dataset(DatasetKind::Completion, "");
        let tok = |s: &str| text.tokenizer.token_to_id(s).unwrap();
        let ex = example(&ds, &text, &json!({ "text": "hi there" })).unwrap();
        assert_eq!(ex.input_ids, [tok("hi"), tok("there"), text.eos_id]);
        assert_eq!(
            ex.labels,
            [tok("hi") as i64, tok("there") as i64, text.eos_id as i64]
        );
        let ex = example(&ds, &text, &json!({ "text": "hi <|im_end|>" })).unwrap();
        assert_eq!(ex.input_ids, [tok("hi"), text.eos_id]);
        ds.field = "content".into();
        assert!(example(&ds, &text, &json!({ "text": "hi" })).is_err());
        let p = prompt(
            &ds,
            &text,
            &json!({ "content": "how are you", "solution": "fine" }),
        )
        .unwrap();
        assert_eq!(
            (p.text.as_str(), p.answer.as_deref(), p.input_ids.len()),
            ("how are you", Some("fine"), 3)
        );
        assert_eq!(
            prompt(&ds, &text, &json!({ "content": "hi" }))
                .unwrap()
                .answer,
            None
        );
    }
}
