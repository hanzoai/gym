use super::*;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

fn design(yaml: &str) -> Design {
    serde_yaml::from_str(yaml).unwrap()
}

fn plan(yaml: &str) -> Result<Plan> {
    Plan::new(design(yaml))
}

fn fails(yaml: &str) -> String {
    match serde_yaml::from_str::<Design>(yaml)
        .map_err(anyhow::Error::from)
        .and_then(Plan::new)
    {
        Ok(_) => panic!("loaded: {yaml}"),
        Err(e) => e.to_string(),
    }
}

fn sampler(yaml: &str) -> Sampler {
    serde_yaml::from_str(yaml).unwrap()
}

fn draws(s: &Sampler, n: usize) -> Vec<Value> {
    let mut rng = rand::rng();
    (0..n)
        .map(|_| s.draw(&Row::new(), &mut rng).unwrap())
        .collect()
}

#[test]
fn category_weights() {
    let s = sampler("{sampler: category, values: [a, b, c], weights: [6, 3, 1]}");
    let v = draws(&s, 4000);
    let share = |x: &str| v.iter().filter(|v| v == &&Value::from(x)).count() as f64 / 4000.0;
    assert!((share("a") - 0.6).abs() < 0.05, "{}", share("a"));
    assert!((share("b") - 0.3).abs() < 0.05, "{}", share("b"));
    assert!((share("c") - 0.1).abs() < 0.05, "{}", share("c"));
    assert!(sampler("{sampler: category, values: [a, b], weights: [1]}")
        .check(None)
        .is_err());
    assert!(
        sampler("{sampler: category, values: [a, b], weights: [0, 0]}")
            .check(None)
            .is_err()
    );
    assert!(sampler("{sampler: category, values: []}")
        .check(None)
        .is_err());
}

#[test]
fn subcategory_binds_to_parent() {
    let parent = sampler("{sampler: category, values: [x, 7]}");
    let s = sampler("{sampler: subcategory, parent: p, values: {x: [x1, x2], '7': [seven]}}");
    s.check(Some(&parent)).unwrap();
    let mut rng = rand::rng();
    let mut row = Row::new();
    row.insert("p".into(), Value::from(7));
    assert_eq!(s.draw(&row, &mut rng).unwrap(), Value::from("seven"));
    row.insert("p".into(), Value::from("x"));
    let v = sample::text(&s.draw(&row, &mut rng).unwrap());
    assert!(v == "x1" || v == "x2");
    row.insert("p".into(), Value::from("q"));
    assert!(s.draw(&row, &mut rng).is_err());
    let partial = sampler("{sampler: subcategory, parent: p, values: {x: [x1]}}");
    assert!(partial
        .check(Some(&parent))
        .unwrap_err()
        .to_string()
        .contains("missing"));
    assert!(s.check(Some(&sampler("{sampler: uuid}"))).is_err());
    assert!(s.check(None).is_err());
}

#[test]
fn uniform_and_gaussian_bounds() {
    let u = sampler("{sampler: uniform, low: 2, high: 5}");
    assert!(draws(&u, 500)
        .iter()
        .all(|v| (2.0..5.0).contains(&v.as_f64().unwrap())));
    let i = sampler("{sampler: uniform, low: 2, high: 5, integer: true}");
    let v = draws(&i, 500);
    assert!(v.iter().all(|v| (2..=5).contains(&v.as_i64().unwrap())));
    assert!(v.contains(&Value::from(5)) && v.contains(&Value::from(2)));
    assert!(sampler("{sampler: uniform, low: 5, high: 5}")
        .check(None)
        .is_err());
    let g = sampler("{sampler: gaussian, mean: 0, std: 1, min: -0.5, max: 0.5}");
    assert!(draws(&g, 500)
        .iter()
        .all(|v| (-0.5..=0.5).contains(&v.as_f64().unwrap())));
    let far = sampler("{sampler: gaussian, mean: 0, std: 1, min: 100}");
    assert!(far.draw(&Row::new(), &mut rand::rng()).is_err());
    assert!(sampler("{sampler: gaussian, mean: 0, std: 0}")
        .check(None)
        .is_err());
    let b = sampler("{sampler: bernoulli, p: 1}");
    assert!(draws(&b, 20).iter().all(|v| v == &Value::from(1)));
}

#[test]
fn datetime_format_and_range() {
    let s = sampler(
        "{sampler: datetime, start: '2024-01-01', end: '2024-01-02', format: '%Y/%m/%d %H'}",
    );
    s.check(None).unwrap();
    for v in draws(&s, 200) {
        assert!(v.as_str().unwrap().starts_with("2024/01/01 "), "{v}");
    }
    let iso =
        sampler("{sampler: datetime, start: '2024-01-01T12:00:00', end: '2024-01-01T12:00:01'}");
    assert_eq!(draws(&iso, 1)[0], Value::from("2024-01-01T12:00:00"));
    assert!(
        sampler("{sampler: datetime, start: '2024-01-02', end: '2024-01-01'}")
            .check(None)
            .is_err()
    );
    assert!(
        sampler("{sampler: datetime, start: '2024-01-01', end: '2024-01-02', format: '%Q'}")
            .check(None)
            .is_err()
    );
}

#[test]
fn uuid_unique_hex() {
    let v = draws(&sampler("{sampler: uuid}"), 1000);
    let set: BTreeSet<&str> = v.iter().map(|v| v.as_str().unwrap()).collect();
    assert_eq!(set.len(), 1000);
    assert!(set
        .iter()
        .all(|s| s.len() == 32 && s.bytes().all(|b| b.is_ascii_hexdigit())));
}

#[test]
fn order_follows_references() {
    let p = plan(
        "columns:
  - {name: full, expression: '{{ first }} {{ last }}'}
  - {name: last, sampler: category, values: [Doe]}
  - {name: first, sampler: category, values: [Jane]}
  - {name: kind, sampler: subcategory, parent: family, values: {a: [a1]}}
  - {name: family, sampler: category, values: [a]}",
    )
    .unwrap();
    assert_eq!(p.order(), ["last", "first", "full", "family", "kind"]);
    assert_eq!(p.downstream("first"), vec![2, 0]);
}

#[test]
fn cycle_and_unknown_reference_fail_at_load() {
    let e = fails(
        "columns:
  - {name: a, expression: '{{ b }}'}
  - {name: b, expression: '{{ a }}'}",
    );
    assert!(e.contains("cycle") && e.contains("a, b"), "{e}");
    let e = fails("columns: [{name: a, expression: '{{ a }}'}]");
    assert!(e.contains("references itself"), "{e}");
    let e = fails("columns: [{name: a, expression: '{{ nope }}'}]");
    assert!(e.contains("`nope`"), "{e}");
    let e = fails("columns: [{name: a, sampler: uuid, extra: 1}]");
    assert!(e.contains("extra"), "{e}");
    let e = fails("columns: [{name: a, llm: text, prompt: hi}]");
    assert!(e.contains("`model`"), "{e}");
    let e = fails("columns: [{name: a, llm: text, prompt: hi, extra: 1}]");
    assert!(e.contains("extra"), "{e}");
    let e = fails("columns: [{name: a, llm: judge, prompt: hi, rubric: {1: bad}, columns: [b]}, {name: b, sampler: uuid}]");
    assert!(e.contains("`model`"), "{e}");
    assert!(serde_yaml::from_str::<Design>("rows: 1, columns: [], foo: 1").is_err());
}

#[test]
fn seed_columns_render_in_templates() {
    let dir = crate::data::fixture::dir("synth-seed");
    let path = dir.join("seed.jsonl");
    std::fs::write(&path, "{\"city\":\"Oslo\"}\n{\"city\":\"Lima\"}\n").unwrap();
    let p = plan(&format!(
        "seed: {{path: '{}'}}
columns:
  - {{name: line, expression: 'from {{{{ city }}}}'}}",
        path.display()
    ))
    .unwrap();
    assert_eq!(p.seeds(5), [0, 1, 0, 1, 0]);
    let mut rng = rand::rng();
    let Outcome::Row(r) = p.row(Some(&p.seed[1]), &mut rng).unwrap() else {
        panic!()
    };
    assert_eq!(r["line"], "from Lima");
    assert_eq!(r["city"], "Lima");
    let s = plan(&format!(
        "seed: {{path: '{}', sampling: shuffle}}\ncolumns: [{{name: x, sampler: uuid}}]",
        path.display()
    ))
    .unwrap();
    let ids = s.seeds(5);
    assert_eq!(ids.len(), 5);
    for pass in ids.chunks(2).take(2) {
        assert_eq!(
            pass.iter().copied().collect::<BTreeSet<_>>(),
            BTreeSet::from([0, 1])
        );
    }
    let e = fails(&format!(
        "seed: {{path: '{}'}}\ncolumns: [{{name: city, sampler: uuid}}]",
        path.display()
    ));
    assert!(e.contains("defined twice"), "{e}");
}

fn check(yaml: &str) -> Check {
    Check::compile(&serde_yaml::from_str(yaml).unwrap()).unwrap()
}

#[test]
fn validators() {
    let mut env = Environment::new();
    env.set_undefined_behavior(UndefinedBehavior::Strict);
    let mut row = Row::new();
    row.insert("a".into(), Value::from("hello world"));
    row.insert("n".into(), Value::from(3));
    row.insert("j".into(), Value::from("{\"k\": 1}"));
    let re = check("{column: a, kind: regex, pattern: '^hello'}");
    assert!(re.passes(&row, &env).unwrap());
    assert!(!check("{column: a, kind: regex, pattern: '^world'}")
        .passes(&row, &env)
        .unwrap());
    assert!(check("{column: a, kind: length, min: 5, max: 11}")
        .passes(&row, &env)
        .unwrap());
    assert!(!check("{column: a, kind: length, max: 10}")
        .passes(&row, &env)
        .unwrap());
    let js = "{column: j, kind: json_schema, json_schema: {type: object, required: [k], properties: {k: {type: integer}}}}";
    assert!(check(js).passes(&row, &env).unwrap());
    row.insert("j".into(), Value::from("{\"k\": \"one\"}"));
    assert!(!check(js).passes(&row, &env).unwrap());
    row.insert("j".into(), json!({"k": 2}));
    assert!(check(js).passes(&row, &env).unwrap());
    assert!(check(
        "{column: n, kind: expression, expression: 'n > 2 and a is startingwith \"hello\"'}"
    )
    .passes(&row, &env)
    .unwrap());
    assert!(!check("{column: n, kind: expression, expression: 'n > 3'}")
        .passes(&row, &env)
        .unwrap());
    assert!(
        check("{column: n, kind: expression, expression: 'zzz > 3'}")
            .passes(&row, &env)
            .is_err()
    );
    assert_eq!(re.on_fail, OnFail::Drop);
    assert_eq!(
        check("{column: a, kind: regex, pattern: x, on_fail: 'retry:2'}").on_fail,
        OnFail::Retry(2)
    );
    assert!(serde_yaml::from_str::<Validator>(
        "{column: a, kind: regex, pattern: x, on_fail: again}"
    )
    .is_err());
    assert!(Check::compile(&serde_yaml::from_str("{column: a, kind: regex}").unwrap()).is_err());
    assert!(Check::compile(
        &serde_yaml::from_str("{column: a, kind: regex, pattern: '('}").unwrap()
    )
    .is_err());
    assert!(Check::compile(&serde_yaml::from_str("{column: a, kind: length}").unwrap()).is_err());
}

#[test]
fn drop_versus_retry() {
    // Whether the row survives depends only on the retry budget: each redraw of
    // `n` has a 1/2 chance of passing, so retry:40 passes and drop never does.
    let p = plan(
        "columns:
  - {name: n, sampler: category, values: [1, 2]}
  - {name: twice, expression: '{{ n * 2 }}'}
validators:
  - {column: n, kind: expression, expression: 'n == 2', on_fail: 'retry:40'}",
    )
    .unwrap();
    let mut rng = rand::rng();
    for _ in 0..20 {
        let Outcome::Row(r) = p.row(None, &mut rng).unwrap() else {
            panic!("dropped")
        };
        assert_eq!(r["n"], 2);
        assert_eq!(r["twice"], "4");
    }
    let p = plan(
        "columns:
  - {name: n, sampler: category, values: [1]}
validators:
  - {column: n, kind: expression, expression: 'n == 2', on_fail: 'retry:3'}
  - {column: n, kind: expression, expression: 'true'}",
    )
    .unwrap();
    assert!(matches!(p.row(None, &mut rng).unwrap(), Outcome::Dropped));
}

#[test]
fn json_object_extraction() {
    assert_eq!(llm::object("{\"a\": 1}").unwrap(), json!({"a": 1}));
    assert_eq!(
        llm::object("Sure!\n```json\n{\"a\": {\"b\": \"}\"}}\n```\nDone.").unwrap(),
        json!({"a": {"b": "}"}})
    );
    assert_eq!(
        llm::object("first {\"a\": 1} then ```json\n{\"a\": 2}\n```").unwrap(),
        json!({"a": 2})
    );
    assert_eq!(
        llm::object("text {\"a\": \"x{y\"} tail").unwrap(),
        json!({"a": "x{y"})
    );
    assert_eq!(
        llm::object("{\"bad\": } {\"ok\": true}").unwrap(),
        json!({"ok": true})
    );
    assert!(llm::object("[1, 2]").is_err());
    assert!(llm::object("no json here").is_err());
}

#[test]
fn judge_parsing() {
    assert_eq!(
        llm::verdict("{\"score\": 4, \"reasoning\": \"fine\"}", 1, 5).unwrap(),
        (4, "fine".into())
    );
    assert_eq!(
        llm::verdict(
            "```json\n{\"score\": \"3.0\", \"reasoning\": \"ok\"}\n```",
            1,
            5
        )
        .unwrap()
        .0,
        3
    );
    assert_eq!(
        llm::verdict("{\"score\": 5.0}", 1, 5).unwrap(),
        (5, String::new())
    );
    assert!(llm::verdict("{\"score\": 6, \"reasoning\": \"\"}", 1, 5).is_err());
    assert!(llm::verdict("{\"score\": 2.5, \"reasoning\": \"\"}", 1, 5).is_err());
    assert!(llm::verdict("{\"reasoning\": \"\"}", 1, 5).is_err());
}

/// A chat-completions server on a random port. The first request is answered
/// 429; afterwards the reply depends on what the prompt asks for.
fn serve() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}/v1", listener.local_addr().unwrap());
    let limited = Arc::new(AtomicBool::new(false));
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let limited = limited.clone();
            std::thread::spawn(move || {
                let mut stream = stream.unwrap();
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let (mut line, mut length, mut auth) = (String::new(), 0usize, false);
                loop {
                    line.clear();
                    reader.read_line(&mut line).unwrap();
                    let lower = line.to_ascii_lowercase();
                    if let Some(n) = lower.strip_prefix("content-length:") {
                        length = n.trim().parse().unwrap();
                    }
                    auth |= lower.starts_with("authorization: bearer test-key");
                    if line == "\r\n" {
                        break;
                    }
                }
                let mut body = vec![0; length];
                reader.read_exact(&mut body).unwrap();
                let req: Value = serde_json::from_slice(&body).unwrap();
                let user = req["messages"].as_array().unwrap().last().unwrap()["content"]
                    .as_str()
                    .unwrap();
                let (status, content) = if !auth {
                    ("401 Unauthorized", "bad key".to_string())
                } else if !limited.swap(true, Relaxed) {
                    ("429 Too Many Requests", "slow down".to_string())
                } else if user.starts_with("Rate") {
                    (
                        "200 OK",
                        "Verdict:\n```json\n{\"score\": 4, \"reasoning\": \"mostly right\"}\n```"
                            .into(),
                    )
                } else if user.starts_with("Extract") {
                    ("200 OK", "{\"lang\": \"en\", \"words\": 5}".into())
                } else if user.starts_with("Answer") {
                    (
                        "200 OK",
                        format!("answer to [{}]", user.lines().last().unwrap()),
                    )
                } else {
                    (
                        "200 OK",
                        format!("question about {}", user.split_whitespace().last().unwrap()),
                    )
                };
                let reply = if status.starts_with("200") {
                    json!({"choices": [{"message": {"role": "assistant", "content": content}}], "usage": {"total_tokens": 7}})
                        .to_string()
                } else {
                    content
                };
                write!(
                    stream,
                    "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{reply}",
                    reply.len()
                )
                .unwrap();
            });
        }
    });
    url
}

#[test]
fn end_to_end_against_a_fake_endpoint() {
    let url = serve();
    let dir = crate::data::fixture::dir("synth-e2e");
    let out = dir.join("out.jsonl");
    let yaml = format!(
        "model: {{endpoint: '{url}', api_key_env: GYM_SYNTH_TEST_KEY, name: fake, concurrency: 3}}
rows: 6
output: '{}'
columns:
  - {{name: domain, sampler: category, values: [finance, health]}}
  - {{name: topic, sampler: subcategory, parent: domain, values: {{finance: [tax], health: [sleep]}}}}
  - {{name: question, llm: text, prompt: 'Write a question about {{{{ topic }}}}', system: 'Terse.'}}
  - {{name: answer, llm: text, prompt: \"Answer this:\\n{{{{ question }}}}\"}}
  - {{name: facts, llm: structured, prompt: 'Extract facts from {{{{ answer }}}}', json_schema: {{type: object, required: [lang, words], properties: {{lang: {{type: string}}, words: {{type: integer}}}}}}}}
  - {{name: quality, llm: judge, prompt: 'Rate the answer.', rubric: {{5: perfect, 3: partial, 1: wrong}}, columns: [question, answer]}}
  - {{name: text, expression: '{{{{ question }}}} / {{{{ answer }}}} / {{{{ facts.lang }}}}'}}
validators:
  - {{column: answer, kind: regex, pattern: '^answer to', on_fail: 'retry:1'}}
  - {{column: quality, kind: expression, expression: 'quality >= 3'}}
  - {{column: facts, kind: json_schema, json_schema: {{type: object}}}}",
        out.display()
    );
    let path = dir.join("design.yml");
    std::fs::write(&path, &yaml).unwrap();
    std::env::set_var("GYM_SYNTH_TEST_KEY", "test-key");
    run(path.to_str().unwrap(), None).unwrap();
    let rows: Vec<Row> = std::fs::read_to_string(&out)
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert_eq!(rows.len(), 6);
    for r in &rows {
        let topic = r["topic"].as_str().unwrap();
        assert!(topic == "tax" || topic == "sleep");
        assert_eq!(r["question"], format!("question about {topic}"));
        assert_eq!(r["answer"], format!("answer to [question about {topic}]"));
        assert_eq!(r["facts"], json!({"lang": "en", "words": 5}));
        assert_eq!(r["quality"], 4);
        assert_eq!(r["quality_reasoning"], "mostly right");
        assert_eq!(
            r["text"],
            format!("question about {topic} / answer to [question about {topic}] / en")
        );
    }
    let plan = Plan::load(path.to_str().unwrap()).unwrap();
    let client = plan.client.as_ref().unwrap();
    assert_eq!(client.calls.load(Relaxed), 0);
    plan.generate(1, Sink::Stdout).unwrap();
    assert_eq!(client.calls.load(Relaxed), 4);
    assert_eq!(client.tokens.load(Relaxed), 28);

    std::env::set_var("GYM_SYNTH_TEST_KEY", "wrong");
    let e = run(path.to_str().unwrap(), Some(1))
        .unwrap_err()
        .to_string();
    assert!(e.contains("401"), "{e}");
}

#[test]
fn parquet_output() {
    let dir = crate::data::fixture::dir("synth-parquet");
    let out = dir.join("out.parquet");
    let p = plan(&format!(
        "rows: 700
output: '{}'
columns:
  - {{name: id, sampler: uuid}}
  - {{name: n, sampler: uniform, low: 0, high: 10, integer: true}}
  - {{name: x, sampler: gaussian, mean: 0, std: 1}}
  - {{name: flag, sampler: bernoulli, p: 0.5}}
  - {{name: nested, expression: '{{{{ n }}}}'}}",
        out.display()
    ))
    .unwrap();
    p.generate(700, Sink::open(out.to_str().unwrap()).unwrap())
        .unwrap();
    let ds =
        serde_yaml::from_str(&format!("{{path: '{}', type: completion}}", out.display())).unwrap();
    let rows = crate::data::source::rows(&ds).unwrap();
    assert_eq!(rows.len(), 700);
    assert!(rows[0]["id"].as_str().unwrap().len() == 32);
    assert!(rows[0]["n"].is_i64() && rows[0]["x"].is_f64() && rows[0]["flag"].is_i64());
    assert_eq!(rows[0]["nested"], rows[0]["n"].to_string());
    assert!(Sink::open(dir.join("out.csv").to_str().unwrap()).is_err());
}
