//! Integration tests for the lomor provider.
//!
//! These tests require a live lomord instance running at
//! `http://localhost:8080` with a generative model loaded.  They are gated
//! behind the `LOMOR_INTEGRATION_TESTS` environment variable so they do not
//! run in normal CI:
//!
//! ```sh
//! LOMOR_INTEGRATION_TESTS=1 cargo test -p codex-lomor -- --test-threads=1
//! ```
//!
//! Run from within the codex devShell:
//!
//! ```sh
//! LOMOR_INTEGRATION_TESTS=1 direnv exec . cargo test -p codex-lomor -- --test-threads=1
//! ```

use std::collections::HashSet;
use std::time::Duration;

const LOMOR_BASE: &str = "http://127.0.0.1:8080";

/// Returns `true` when `LOMOR_INTEGRATION_TESTS` is set to a non-empty value.
fn integration_enabled() -> bool {
    std::env::var("LOMOR_INTEGRATION_TESTS")
        .map(|v| !v.is_empty() && v != "0")
        .unwrap_or(false)
}

// ── Health check ───────────────────────────────────────────────────────────────

/// Confirms lomord responds to `GET /health` with 200.
///
/// This is the cheapest possible liveness check — it passes even when no
/// model is loaded.
#[tokio::test]
async fn health_endpoint_returns_200() {
    if !integration_enabled() {
        return;
    }

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
        .expect("client build");

    let resp = client
        .get(format!("{LOMOR_BASE}/health"))
        .send()
        .await
        .expect("GET /health");

    assert!(
        resp.status().is_success(),
        "GET /health returned {}",
        resp.status()
    );
}

// ── Models list ────────────────────────────────────────────────────────────────

/// Confirms `GET /v1/models` returns at least one model entry.
#[tokio::test]
async fn models_list_not_empty() {
    if !integration_enabled() {
        return;
    }

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
        .expect("client build");

    let resp = client
        .get(format!("{LOMOR_BASE}/v1/models"))
        .send()
        .await
        .expect("GET /v1/models")
        .error_for_status()
        .expect("GET /v1/models returned error status");

    let body: serde_json::Value = resp.json().await.expect("parse models JSON");
    let models = body["data"].as_array().expect("data array");
    assert!(!models.is_empty(), "expected at least one loaded model");

    // Print the available models for debugging.
    for m in models {
        eprintln!("available model: {}", m["id"].as_str().unwrap_or("?"));
    }
}

// ── Responses API stream ───────────────────────────────────────────────────────

/// Sends a minimal `POST /v1/responses` request and asserts that the SSE
/// stream contains the four required event types in order.
///
/// The prompt is intentionally simple ("Reply with the single word: OK") so
/// the test completes quickly regardless of model.
#[tokio::test]
async fn responses_api_streams_required_events() {
    if !integration_enabled() {
        return;
    }

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(120))
        .build()
        .expect("client build");

    // Determine which model to use from the models list, preferring the
    // configured codex model from the environment if set.
    let model = get_model(&client).await;
    eprintln!("using model: {model}");

    let body = serde_json::json!({
        "model": model,
        "instructions": "You are a test assistant. Be as brief as possible.",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": "Reply with the single word: OK"
            }
        ],
        "stream": true
    });

    let resp = client
        .post(format!("{LOMOR_BASE}/v1/responses"))
        .header("content-type", "application/json")
        .json(&body)
        .send()
        .await
        .expect("POST /v1/responses");

    assert!(
        resp.status().is_success(),
        "POST /v1/responses returned {}",
        resp.status()
    );
    assert_eq!(
        resp.headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or(""),
        "text/event-stream",
        "expected SSE content-type"
    );

    // Collect the raw SSE body.
    let body_bytes = resp
        .bytes()
        .await
        .expect("read response body");
    let raw = String::from_utf8_lossy(&body_bytes);
    eprintln!("--- raw SSE stream ---\n{raw}\n--- end ---");

    // Parse SSE event types.
    let event_types = parse_sse_event_types(&raw);
    eprintln!("event types received: {event_types:?}");

    // Required events — must all appear.
    let required: &[&str] = &[
        "response.created",
        "response.output_text.delta",
        "response.output_item.done",
        "response.completed",
    ];
    for &ev in required {
        assert!(
            event_types.contains(ev),
            "missing required SSE event type: {ev}"
        );
    }

    // The output must contain at least one delta with non-empty text.
    assert!(
        raw.contains("\"delta\""),
        "expected at least one output_text.delta with a delta field"
    );

    // response.created must come before response.completed.
    let created_pos = raw.find("response.created").expect("response.created position");
    let completed_pos = raw.find("response.completed").expect("response.completed position");
    assert!(
        created_pos < completed_pos,
        "response.created must appear before response.completed"
    );
}

/// Verifies that `POST /v1/responses` with an empty `input` array returns 400.
#[tokio::test]
async fn responses_api_rejects_empty_input() {
    if !integration_enabled() {
        return;
    }

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(10))
        .build()
        .expect("client build");

    let model = get_model(&client).await;

    let body = serde_json::json!({
        "model": model,
        "input": [],
        "stream": true
    });

    let resp = client
        .post(format!("{LOMOR_BASE}/v1/responses"))
        .header("content-type", "application/json")
        .json(&body)
        .send()
        .await
        .expect("POST /v1/responses");

    assert_eq!(
        resp.status(),
        reqwest::StatusCode::BAD_REQUEST,
        "expected 400 for empty input, got {}",
        resp.status()
    );
}

// ── ensure_oss_ready ───────────────────────────────────────────────────────────

/// Calls `ensure_oss_ready` when lomord is already running and asserts it
/// returns without error.
///
/// This is a no-op happy-path test — it does not start anything.
#[tokio::test]
async fn ensure_oss_ready_noop_when_already_running() {
    if !integration_enabled() {
        return;
    }

    codex_lomor::ensure_oss_ready()
        .await
        .expect("ensure_oss_ready should succeed when server is already up");
}

// ── Helpers ────────────────────────────────────────────────────────────────────

/// Returns the model name to use in tests.
///
/// Uses `LOMOR_TEST_MODEL` env var if set, otherwise picks the first model
/// returned by `GET /v1/models`, otherwise falls back to `"qwen3.6-35b-smart"`.
async fn get_model(client: &reqwest::Client) -> String {
    if let Ok(m) = std::env::var("LOMOR_TEST_MODEL") {
        if !m.is_empty() {
            return m;
        }
    }
    // Try to pick the first loaded model from the API.
    if let Ok(resp) = client
        .get(format!("{LOMOR_BASE}/v1/models"))
        .timeout(Duration::from_secs(5))
        .send()
        .await
    {
        if let Ok(json) = resp.json::<serde_json::Value>().await {
            if let Some(id) = json["data"]
                .as_array()
                .and_then(|a| a.first())
                .and_then(|m| m["id"].as_str())
            {
                return id.to_owned();
            }
        }
    }
    "qwen3.6-35b-smart".to_owned()
}

/// Extracts all `event: <type>` lines from a raw SSE response body.
fn parse_sse_event_types(raw: &str) -> HashSet<&str> {
    raw.lines()
        .filter_map(|line| line.strip_prefix("event: "))
        .collect()
}

// ── Unit tests (always run) ────────────────────────────────────────────────────

#[test]
fn parse_sse_event_types_extracts_all_types() {
    let raw = "event: response.created\ndata: {}\n\n\
               event: response.output_text.delta\ndata: {}\n\n\
               event: response.completed\ndata: {}\n\n";
    let types = parse_sse_event_types(raw);
    assert!(types.contains("response.created"));
    assert!(types.contains("response.output_text.delta"));
    assert!(types.contains("response.completed"));
    assert!(!types.contains("bogus"));
}

#[test]
fn parse_sse_event_types_empty_on_no_events() {
    let types = parse_sse_event_types("data: {}\n\n");
    assert!(types.is_empty());
}
