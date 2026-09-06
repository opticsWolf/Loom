//! Typed validation channel: errors block, warnings flow.
//!
//! Replaces the `warning:`-prefix string convention with a typed channel;
//! the prefix is re-attached at the Python boundary for compatibility.

/// Severity of a validation finding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Severity {
  Error,
  Warning,
}

/// One validation finding: severity + human message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidationIssue {
  pub severity: Severity,
  pub message: String,
}

impl ValidationIssue {
  /// Blocking finding.
  pub fn error(message: impl Into<String>) -> Self {
    Self { severity: Severity::Error, message: message.into() }
  }

  /// Non-blocking finding (flows to the caller, never blocks).
  pub fn warning(message: impl Into<String>) -> Self {
    Self { severity: Severity::Warning, message: message.into() }
  }

  pub fn is_error(&self) -> bool {
    self.severity == Severity::Error
  }

  /// Split a finding list into `(errors, warnings)`.
  pub fn partition(issues: &[ValidationIssue]) -> (Vec<&ValidationIssue>, Vec<&ValidationIssue>) {
    issues.iter().partition(|i| i.is_error())
  }

  /// Solve-gate: `Ok` when no errors (warnings pass through silently here;
  /// the caller re-emits them); `Err` joining all error messages.
  pub fn gate(issues: &[ValidationIssue], what: &str) -> Result<(), String> {
    let (errors, _) = Self::partition(issues);
    if errors.is_empty() {
      return Ok(());
    }
    Err(format!(
      "{what}: {} blocking issue(s): {}",
      errors.len(),
      errors.iter().map(|e| e.message.as_str()).collect::<Vec<_>>().join("; ")
    ))
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn gate_blocks_errors_passes_warnings() {
    let clean = vec![ValidationIssue::warning("overhang")];
    assert!(ValidationIssue::gate(&clean, "solve").is_ok());
    let bad = vec![
      ValidationIssue::warning("overhang"),
      ValidationIssue::error("negative thickness"),
    ];
    let err = ValidationIssue::gate(&bad, "solve").unwrap_err();
    assert!(err.contains("negative thickness"));
    assert!(!err.contains("overhang"));
  }
}
