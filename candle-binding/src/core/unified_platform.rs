//! Platform gate for unified mmBERT classifier preparation.
//!
//! Issue #2440: on macOS arm64, preparing the unified mmBERT classifiers can
//! deadlock with every thread parked in `__psynch_cvwait`. A caller timeout
//! cannot unblock that native wait, so initialization is refused before any
//! weight file is opened.

use std::path::{Component, Path, PathBuf};

pub const UNIFIED_MMBERT_REFUSAL: &str =
    "unified mmBERT classifier initialization refused before model preparation on darwin/arm64";

/// `os` and `arch` use Rust consts (`macos`, `aarch64`), not Go's `darwin` / `arm64`.
/// `architecture` is the caller-supplied name. Model paths are consulted only
/// through `config.json`; `model.safetensors` is never opened.
pub fn unified_mmbert_refusal(
    os: &str,
    arch: &str,
    architecture: &str,
    model_paths: &[&str],
) -> Option<&'static str> {
    if os != "macos" || arch != "aarch64" {
        return None;
    }
    if is_unified_mmbert_family(architecture) {
        return Some(UNIFIED_MMBERT_REFUSAL);
    }
    for path in model_paths {
        if config_is_mmbert(path) {
            return Some(UNIFIED_MMBERT_REFUSAL);
        }
    }
    None
}

pub const LEGACY_ENGINE_CONFLICT: &str =
    "unified LoRA classifier already initialized with a different model configuration";

/// Configuration that `ParallelLoRAEngine::new` actually receives.
/// `architecture` is not included: the engine never sees that string, and
/// model behavior comes from each checkpoint's config. Paths are normalized
/// so a relative path and its absolute form are the same engine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LegacyEngineIdentity {
    pub intent_path: String,
    pub pii_path: String,
    pub security_path: String,
    pub use_cpu: bool,
}

impl LegacyEngineIdentity {
    pub fn from_request(intent: &str, pii: &str, security: &str, use_cpu: bool) -> Self {
        Self {
            intent_path: normalize_model_path(intent),
            pii_path: normalize_model_path(pii),
            security_path: normalize_model_path(security),
            use_cpu,
        }
    }
}

/// Absolute form of a model directory. An existing path is canonicalized so
/// symlinks and `.` agree. A missing path is still made absolute, so
/// `models/intent` and `<cwd>/models/intent` compare equal before the files exist.
pub fn normalize_model_path(path: &str) -> String {
    let path = Path::new(path);
    if let Ok(canonical) = path.canonicalize() {
        return canonical.to_string_lossy().into_owned();
    }
    lexical_absolute(path).to_string_lossy().into_owned()
}

fn lexical_absolute(path: &Path) -> PathBuf {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .map(|cwd| cwd.join(path))
            .unwrap_or_else(|_| path.to_path_buf())
    };
    let mut normalized = PathBuf::new();
    for component in absolute.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                normalized.pop();
            }
            other => normalized.push(other.as_os_str()),
        }
    }
    normalized
}

/// What the legacy FFI initializer does next. A platform refusal wins over an
/// already-stored engine. A stored engine is reused only when its identity
/// matches the request; a different model is a conflict, not a success.
#[derive(Debug, PartialEq, Eq)]
pub enum LegacyInitStep {
    Refuse,
    ReuseExisting,
    RejectConflict,
    Prepare,
}

pub fn legacy_init_step(
    stored: Option<&LegacyEngineIdentity>,
    requested: &LegacyEngineIdentity,
    refusal: Option<&str>,
) -> LegacyInitStep {
    if refusal.is_some() {
        return LegacyInitStep::Refuse;
    }
    match stored {
        Some(existing) if existing == requested => LegacyInitStep::ReuseExisting,
        Some(_) => LegacyInitStep::RejectConflict,
        None => LegacyInitStep::Prepare,
    }
}

/// Result of publishing a newly built engine into the process-wide slot.
/// Failure is success only when the engine already stored is this request.
/// Another caller's model must not be reported as this caller's initialization.
pub fn legacy_publish_result(published: bool, stored_matches_request: bool) -> bool {
    published || stored_matches_request
}

/// Explicit adapter names. `"modernbert"` is not mmBERT; both families use that
/// `model_type`, and `ModernBertVariant` separates them by vocabulary size.
pub fn is_unified_mmbert_family(name: &str) -> bool {
    matches!(
        name.trim().to_ascii_lowercase().as_str(),
        "mmbert" | "mmbert32k" | "mmbert-32k"
    )
}

/// Same signal as `ModernBertVariant::detect_from_config`: mmBERT has
/// `vocab_size >= 200000` and `position_embedding_type == "sans_pos"`.
/// Missing or unreadable config is not a refusal. Weights are never opened.
fn config_is_mmbert(model_path: &str) -> bool {
    let data = match std::fs::read_to_string(Path::new(model_path).join("config.json")) {
        Ok(data) => data,
        Err(_) => return false,
    };
    let value: serde_json::Value = match serde_json::from_str(&data) {
        Ok(value) => value,
        Err(_) => return false,
    };
    if value
        .get("model_type")
        .and_then(|item| item.as_str())
        .is_some_and(is_unified_mmbert_family)
    {
        return true;
    }
    let vocab = value
        .get("vocab_size")
        .and_then(|item| item.as_u64())
        .unwrap_or(0);
    let position = value
        .get("position_embedding_type")
        .and_then(|item| item.as_str())
        .unwrap_or("");
    vocab >= 200_000 && position == "sans_pos"
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn refuses_mmbert_family_on_macos_arm64_without_weights() {
        for name in ["mmbert", "mmbert32k", "mmbert-32k"] {
            assert_eq!(
                unified_mmbert_refusal("macos", "aarch64", name, &[]),
                Some(UNIFIED_MMBERT_REFUSAL),
                "{name}"
            );
        }
        assert_eq!(
            unified_mmbert_refusal("macos", "aarch64", "modernbert", &[]),
            None,
            "ordinary ModernBERT must stay available"
        );
    }

    fn identity(intent: &str, use_cpu: bool) -> LegacyEngineIdentity {
        LegacyEngineIdentity {
            intent_path: intent.to_string(),
            pii_path: "pii".to_string(),
            security_path: "security".to_string(),
            use_cpu,
        }
    }

    #[test]
    fn mmbert_refusal_beats_an_existing_engine() {
        let stored = identity("intent", true);
        assert_eq!(
            legacy_init_step(Some(&stored), &stored, Some(UNIFIED_MMBERT_REFUSAL)),
            LegacyInitStep::Refuse
        );
        assert_eq!(
            legacy_init_step(Some(&stored), &stored, None),
            LegacyInitStep::ReuseExisting
        );
        assert_eq!(
            legacy_init_step(None, &stored, None),
            LegacyInitStep::Prepare
        );
    }

    #[test]
    fn conflicting_second_request_is_not_success() {
        let stored = identity("intent-a", true);
        let other_model = identity("intent-b", true);
        let other_device = identity("intent-a", false);
        assert_eq!(
            legacy_init_step(Some(&stored), &other_model, None),
            LegacyInitStep::RejectConflict
        );
        assert_eq!(
            legacy_init_step(Some(&stored), &other_device, None),
            LegacyInitStep::RejectConflict
        );
        assert!(
            !legacy_publish_result(false, false),
            "a lost race against a different model is not success"
        );
        assert!(
            legacy_publish_result(false, true),
            "the same configuration already stored is success"
        );
        assert!(legacy_publish_result(true, true));
    }

    #[test]
    fn relative_and_absolute_paths_are_the_same_engine() {
        let dir = tempfile::tempdir().unwrap();
        let absolute = dir.path().canonicalize().unwrap();
        let dotted = absolute.join(".");
        let slashed = format!("{}/", absolute.display());
        assert_eq!(
            normalize_model_path(absolute.to_str().unwrap()),
            normalize_model_path(dotted.to_str().unwrap())
        );
        assert_eq!(
            normalize_model_path(absolute.to_str().unwrap()),
            normalize_model_path(&slashed)
        );

        let missing_name = "legacy-engine-identity-missing";
        let relative = normalize_model_path(missing_name);
        let from_cwd = std::env::current_dir().unwrap().join(missing_name);
        assert_eq!(relative, normalize_model_path(from_cwd.to_str().unwrap()));
        assert_eq!(relative, normalize_model_path(&format!("./{missing_name}")));

        let stored =
            LegacyEngineIdentity::from_request(absolute.to_str().unwrap(), "pii", "security", true);
        let requested = LegacyEngineIdentity::from_request(&slashed, "pii", "security", true);
        assert_eq!(
            legacy_init_step(Some(&stored), &requested, None),
            LegacyInitStep::ReuseExisting
        );
    }

    #[test]
    fn trailing_space_and_backslash_are_different_checkpoints() {
        let parent = tempfile::tempdir().unwrap();
        let plain = parent.path().join("a");
        let spaced = parent.path().join("a ");
        let backslash = parent.path().join("a\\");
        std::fs::create_dir(&plain).unwrap();
        std::fs::create_dir(&spaced).unwrap();
        std::fs::create_dir(&backslash).unwrap();

        let plain_norm = normalize_model_path(plain.to_str().unwrap());
        assert_ne!(plain_norm, normalize_model_path(spaced.to_str().unwrap()));
        assert_ne!(
            plain_norm,
            normalize_model_path(backslash.to_str().unwrap())
        );

        let stored =
            LegacyEngineIdentity::from_request(plain.to_str().unwrap(), "pii", "sec", true);
        let spaced_request =
            LegacyEngineIdentity::from_request(spaced.to_str().unwrap(), "pii", "sec", true);
        assert_eq!(
            legacy_init_step(Some(&stored), &spaced_request, None),
            LegacyInitStep::RejectConflict
        );
    }

    #[test]
    fn allows_other_platforms_and_bert() {
        assert_eq!(
            unified_mmbert_refusal("linux", "aarch64", "mmbert32k", &[]),
            None
        );
        assert_eq!(
            unified_mmbert_refusal("macos", "x86_64", "modernbert", &[]),
            None
        );
        assert_eq!(
            unified_mmbert_refusal("macos", "aarch64", "bert", &[]),
            None
        );
        assert_eq!(
            unified_mmbert_refusal("macos", "aarch64", "bert_lora", &[]),
            None
        );
    }

    #[test]
    fn reads_config_json_and_does_not_require_weights() {
        let dir = tempfile::tempdir().unwrap();
        let config_path = dir.path().join("config.json");
        let mut config = std::fs::File::create(&config_path).unwrap();
        write!(
            config,
            r#"{{"model_type":"modernbert","vocab_size":256000,"position_embedding_type":"sans_pos","max_position_embeddings":32768}}"#
        )
        .unwrap();
        assert!(
            !dir.path().join("model.safetensors").exists(),
            "weights must stay unread and absent"
        );

        let path = dir.path().to_str().unwrap();
        assert_eq!(
            unified_mmbert_refusal("macos", "aarch64", "modernbert", &[path, path, path]),
            Some(UNIFIED_MMBERT_REFUSAL)
        );
        assert_eq!(
            unified_mmbert_refusal("linux", "aarch64", "", &[path]),
            None
        );

        let mut ordinary = std::fs::File::create(&config_path).unwrap();
        write!(
            ordinary,
            r#"{{"model_type":"modernbert","vocab_size":50368,"position_embedding_type":"sans_pos","max_position_embeddings":8192}}"#
        )
        .unwrap();
        assert_eq!(
            unified_mmbert_refusal("macos", "aarch64", "modernbert", &[path]),
            None
        );
    }
}
