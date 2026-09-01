//! The tokenizer, its special ids, and the chat template a run renders with.

use crate::hub;
use crate::Config;
use anyhow::{anyhow, Context, Result};
use hf_hub::api::sync::Api;
use hf_hub::{Repo, RepoType};
use minijinja::{context, Environment, Error, ErrorKind};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;

/// One chat turn as the template sees it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

pub struct Text {
    pub tokenizer: Tokenizer,
    pub template: Option<String>,
    pub eos_id: u32,
    pub pad_id: u32,
    pub bos_id: Option<u32>,
}

impl Text {
    pub fn load(cfg: &Config) -> Result<Text> {
        let (tok, tc) = match &cfg.tokenizer_config {
            Some(src) => files(src)?,
            None => {
                let s = hub::snapshot(&cfg.base_model)?;
                (s.tokenizer, s.tokenizer_config)
            }
        };
        let tokenizer =
            Tokenizer::from_file(&tok).map_err(|e| anyhow!("{}: {e}", tok.display()))?;
        let tc: Value = match tc {
            Some(p) => serde_json::from_slice(&std::fs::read(&p)?)
                .with_context(|| p.display().to_string())?,
            None => Value::Null,
        };
        let id = |k: &str| token(&tc[k]).and_then(|t| tokenizer.token_to_id(t));
        let eos_id = id("eos_token").context("tokenizer_config.json: eos_token")?;
        Ok(Text {
            template: template(cfg.chat_template.as_deref(), &tc),
            eos_id,
            pad_id: id("pad_token").unwrap_or(eos_id),
            bos_id: id("bos_token"),
            tokenizer,
        })
    }

    pub fn encode(&self, s: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        Ok(self
            .tokenizer
            .encode(s, add_special_tokens)
            .map_err(|e| anyhow!("{e}"))?
            .get_ids()
            .to_vec())
    }
}

/// `tokenizer.json` and `tokenizer_config.json` from a directory or a model repo.
fn files(src: &str) -> Result<(PathBuf, Option<PathBuf>)> {
    let p = Path::new(src);
    if p.is_dir() {
        return Ok((
            p.join("tokenizer.json"),
            Some(p.join("tokenizer_config.json")).filter(|p| p.exists()),
        ));
    }
    let repo = Api::new()?.repo(Repo::new(src.to_string(), RepoType::Model));
    Ok((
        repo.get("tokenizer.json").context("tokenizer.json")?,
        repo.get("tokenizer_config.json").ok(),
    ))
}

/// A special token entry: a string or `{"content": ...}`.
fn token(v: &Value) -> Option<&str> {
    v.as_str().or_else(|| v["content"].as_str())
}

/// `None` or `tokenizer_default` takes the tokenizer's own template; a known
/// name takes the built-in; `tokenizer_default_fallback_<name>` prefers the
/// tokenizer's and falls back to the name; anything else is inline Jinja.
fn template(choice: Option<&str>, tc: &Value) -> Option<String> {
    let own = || match &tc["chat_template"] {
        Value::String(s) => Some(s.clone()),
        Value::Array(a) => a
            .iter()
            .find(|t| t["name"] == "default")
            .and_then(|t| t["template"].as_str().map(String::from)),
        _ => None,
    };
    let named = |n: &str| Some(builtin(n).unwrap_or(n).to_string());
    match choice {
        None | Some("tokenizer_default") => own(),
        Some(c) => match c.strip_prefix("tokenizer_default_fallback_") {
            Some(n) => own().or_else(|| named(n)),
            None => named(c),
        },
    }
}

fn builtin(name: &str) -> Option<&'static str> {
    match name {
        "chatml" => Some(CHATML),
        "qwen3" => Some(QWEN3),
        "llama3" => Some(LLAMA3),
        _ => None,
    }
}

/// Render `messages` through the chat template with Hugging Face's variables:
/// `messages`, `add_generation_prompt`, `bos_token`, `eos_token`,
/// `enable_thinking` (false) and a `raise_exception` function.
pub fn render(text: &Text, messages: &[Message], add_generation_prompt: bool) -> Result<String> {
    let src = text
        .template
        .as_deref()
        .context("no chat template: set `chat_template` or use a tokenizer that has one")?;
    let mut env = Environment::new();
    env.set_trim_blocks(true);
    env.set_lstrip_blocks(true);
    env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
    env.add_function("raise_exception", |msg: String| -> Result<(), Error> {
        Err(Error::new(ErrorKind::InvalidOperation, msg))
    });
    let tok = |id: Option<u32>| {
        id.and_then(|i| text.tokenizer.id_to_token(i))
            .unwrap_or_default()
    };
    Ok(env.template_from_str(src)?.render(context! {
        messages,
        add_generation_prompt,
        bos_token => tok(text.bos_id),
        eos_token => tok(Some(text.eos_id)),
        enable_thinking => false,
    })?)
}

pub const CHATML: &str = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}";

pub const LLAMA3: &str = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}";

pub const QWEN3: &str = r##"{%- if tools %}
    {{- '<|im_start|>system\n' }}
    {%- if messages[0].role == 'system' %}
        {{- messages[0].content + '\n\n' }}
    {%- endif %}
    {{- "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>" }}
    {%- for tool in tools %}
        {{- "\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n" }}
{%- else %}
    {%- if messages[0].role == 'system' %}
        {{- '<|im_start|>system\n' + messages[0].content + '<|im_end|>\n' }}
    {%- endif %}
{%- endif %}
{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}
{%- for message in messages[::-1] %}
    {%- set index = (messages|length - 1) - loop.index0 %}
    {%- if ns.multi_step_tool and message.role == "user" and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}
        {%- set ns.multi_step_tool = false %}
        {%- set ns.last_query_index = index %}
    {%- endif %}
{%- endfor %}
{%- for message in messages %}
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) %}
        {{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>' + '\n' }}
    {%- elif message.role == "assistant" %}
        {%- set content = message.content %}
        {%- set reasoning_content = '' %}
        {%- if message.reasoning_content is defined and message.reasoning_content is not none %}
            {%- set reasoning_content = message.reasoning_content %}
        {%- else %}
            {%- if '</think>' in message.content %}
                {%- set content = message.content.split('</think>')[-1].lstrip('\n') %}
                {%- set reasoning_content = message.content.split('</think>')[0].rstrip('\n').split('<think>')[-1].lstrip('\n') %}
            {%- endif %}
        {%- endif %}
        {%- if loop.index0 > ns.last_query_index %}
            {%- if loop.last or (not loop.last and reasoning_content) %}
                {{- '<|im_start|>' + message.role + '\n<think>\n' + reasoning_content.strip('\n') + '\n</think>\n\n' + content.lstrip('\n') }}
            {%- else %}
                {{- '<|im_start|>' + message.role + '\n' + content }}
            {%- endif %}
        {%- else %}
            {{- '<|im_start|>' + message.role + '\n' + content }}
        {%- endif %}
        {%- if message.tool_calls %}
            {%- for tool_call in message.tool_calls %}
                {%- if (loop.first and content) or (not loop.first) %}
                    {{- '\n' }}
                {%- endif %}
                {%- if tool_call.function %}
                    {%- set tool_call = tool_call.function %}
                {%- endif %}
                {{- '<tool_call>\n{"name": "' }}
                {{- tool_call.name }}
                {{- '", "arguments": ' }}
                {%- if tool_call.arguments is string %}
                    {{- tool_call.arguments }}
                {%- else %}
                    {{- tool_call.arguments | tojson }}
                {%- endif %}
                {{- '}\n</tool_call>' }}
            {%- endfor %}
        {%- endif %}
        {{- '<|im_end|>\n' }}
    {%- elif message.role == "tool" %}
        {%- if loop.first or (messages[loop.index0 - 1].role != "tool") %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '\n<tool_response>\n' }}
        {{- message.content }}
        {{- '\n</tool_response>' }}
        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}
            {{- '<|im_end|>\n' }}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
    {%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>\n\n</think>\n\n' }}
    {%- else %}
        {{- '<think>\n\n' }}
    {%- endif %}
{%- endif %}"##;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::fixture;

    fn msgs(v: &[(&str, &str)]) -> Vec<Message> {
        v.iter()
            .map(|(r, c)| Message {
                role: r.to_string(),
                content: c.to_string(),
            })
            .collect()
    }

    #[test]
    fn chatml_render() {
        let t = fixture::text();
        let m = msgs(&[
            ("system", "be good"),
            ("user", "hi there"),
            ("assistant", "fine thanks"),
        ]);
        assert_eq!(
            render(&t, &m, false).unwrap(),
            "<|im_start|>system\nbe good<|im_end|>\n<|im_start|>user\nhi there<|im_end|>\n<|im_start|>assistant\nfine thanks<|im_end|>\n"
        );
        assert!(render(&t, &m[..2], true)
            .unwrap()
            .ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn qwen3_render_thinking_and_generation_prompt() {
        let mut t = fixture::text();
        t.template = Some(QWEN3.into());
        let m = msgs(&[
            ("user", "hi there"),
            ("assistant", "<think>\nponder\n</think>\n\nfine thanks"),
        ]);
        assert_eq!(
            render(&t, &m, false).unwrap(),
            "<|im_start|>user\nhi there<|im_end|>\n<|im_start|>assistant\n<think>\nponder\n</think>\n\nfine thanks<|im_end|>\n"
        );
        assert_eq!(
            render(&t, &m[..1], true).unwrap(),
            "<|im_start|>user\nhi there<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
        let m = msgs(&[
            ("user", "hi"),
            ("assistant", "one"),
            ("user", "two"),
            ("assistant", "three"),
        ]);
        assert_eq!(
            render(&t, &m, false).unwrap(),
            "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\none<|im_end|>\n<|im_start|>user\ntwo<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nthree<|im_end|>\n"
        );
    }

    #[test]
    fn llama3_render_uses_bos() {
        let mut t = fixture::text();
        t.template = Some(LLAMA3.into());
        t.bos_id = t.tokenizer.token_to_id("<|endoftext|>");
        let m = msgs(&[("user", " hi "), ("assistant", "fine")]);
        assert_eq!(
            render(&t, &m, true).unwrap(),
            "<|endoftext|><|start_header_id|>user<|end_header_id|>\n\nhi<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nfine<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        );
    }

    #[test]
    fn raise_exception_is_an_error() {
        let mut t = fixture::text();
        t.template = Some("{{ raise_exception('nope') }}".into());
        assert!(render(&t, &[], false)
            .unwrap_err()
            .to_string()
            .contains("nope"));
    }

    #[test]
    fn template_resolution() {
        let tc = serde_json::json!({ "chat_template": "own" });
        assert_eq!(template(None, &tc).as_deref(), Some("own"));
        assert_eq!(
            template(Some("tokenizer_default"), &tc).as_deref(),
            Some("own")
        );
        assert_eq!(template(Some("chatml"), &tc).as_deref(), Some(CHATML));
        assert_eq!(template(Some("{{ x }}"), &tc).as_deref(), Some("{{ x }}"));
        assert_eq!(
            template(Some("tokenizer_default_fallback_chatml"), &tc).as_deref(),
            Some("own")
        );
        let list = serde_json::json!({ "chat_template": [{ "name": "rag", "template": "r" }, { "name": "default", "template": "d" }] });
        assert_eq!(template(None, &list).as_deref(), Some("d"));
        assert_eq!(template(None, &Value::Null), None);
        assert_eq!(
            template(Some("tokenizer_default_fallback_chatml"), &Value::Null).as_deref(),
            Some(CHATML)
        );
        assert_eq!(token(&serde_json::json!({ "content": "<s>" })), Some("<s>"));
        assert_eq!(token(&serde_json::json!("<s>")), Some("<s>"));
        assert_eq!(token(&Value::Null), None);
    }
}
