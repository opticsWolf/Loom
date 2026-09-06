import io
p = 'rust/navette/src/config.rs'
s = io.open(p, encoding='utf-8').read()
anchor = '/// Everything a program file restores (absent sections stay empty).'
assert anchor in s
block = '''/// Authoring-time validation (the pydantic bound inventory, natively).
/// Called by the PyO3 config constructors; assembly assumes validity.
impl MaterialDef {
  pub fn validate(&self) -> Result<(), String> {
    const MODELS: [&str; 6] = [
      "Konstant", "TableMaterial", "Cauchy", "CauchyUrbach", "Sellmeier", "SellmeierUrbach",
    ];
    if !MODELS.contains(&self.model.as_str()) {
      return Err(format!("Unknown model: {}", self.model));
    }
    let num = |key: &str| -> Result<f64, String> {
      self
        .params
        .get(key)
        .and_then(|v| v.as_f64())
        .ok_or_else(|| format!("{}: '{}' must be a number.", self.model, key))
    };
    let pos = |key: &str| -> Result<(), String> {
      if num(key)? <= 0.0 {
        return Err(format!("{}: '{}' must be > 0.", self.model, key));
      }
      Ok(())
    };
    let nonneg = |key: &str, default: f64| -> Result<(), String> {
      let v = self.params.get(key).and_then(|v| v.as_f64()).unwrap_or(default);
      if v < 0.0 {
        return Err(format!("{}: '{}' must be >= 0.", self.model, key));
      }
      Ok(())
    };
    match self.model.as_str() {
      "Konstant" => {
        pos("n")?;
        nonneg("k", 0.0)?;
      }
      "TableMaterial" => {
        if self.n_data.is_none() {
          return Err("TableMaterial requires n_data".to_string());
        }
        nonneg("n_factor", 1.0)?;
        nonneg("k_factor", 1.0)?;
      }
      "Cauchy" => {
        num("A")?;
        num("B")?;
        num("C")?;
      }
      "CauchyUrbach" => {
        num("A")?;
        num("B")?;
        num("C")?;
        pos("alpha0")?;
        pos("Eu")?;
        pos("lambda_g")?;
      }
      "Sellmeier" => {
        pos("B1")?;
        pos("C1")?;
        pos("B2")?;
        pos("C2")?;
        nonneg("B3", 0.0)?;
        nonneg("C3", 0.0)?;
      }
      _ => {
        pos("B1")?;
        pos("C1")?;
        pos("B2")?;
        pos("C2")?;
        nonneg("B3", 0.0)?;
        nonneg("C3", 0.0)?;
        pos("alpha0")?;
        pos("Eu")?;
        pos("lambda_g")?;
      }
    }
    Ok(())
  }
}

impl LayerRow {
  pub fn validate(&self) -> Result<(), String> {
    if !(self.thickness_nm > 0.0) {
      return Err("thickness_nm must be > 0.".to_string());
    }
    if self.roughness_nm < 0.0 {
      return Err("roughness_nm must be >= 0.".to_string());
    }
    crate::structure::RoughnessType::try_from_i32(self.rough_type)
      .map_err(|e| format!("rough_type: {e}"))?;
    if !(0.0..=1.0).contains(&self.inh_delta) {
      return Err("inh_delta must be in [0, 1].".to_string());
    }
    if self.interface_thickness_nm < 0.0 {
      return Err("interface_thickness_nm must be >= 0.".to_string());
    }
    match self.layer_type {
      0 | 1 | 2 => {}
      t => return Err(format!("layer_type must be 0, 1 or 2 (got {t}).")),
    }
    Ok(())
  }
}

impl GroupRow {
  pub fn validate(&self) -> Result<(), String> {
    if self.error_mask.len() != 6 {
      return Err(format!("group {:?}: error_mask needs 6 entries.", self.name));
    }
    if self.optimization_mask.len() != 7
      || self.optimization_mask.iter().any(|x| *x != 0 && *x != 1)
    {
      return Err(format!("group {:?}: optimization_mask must be 7 binary entries.", self.name));
    }
    for (key, v) in [
      ("thickness", self.thickness_error_type),
      ("n", self.n_error_type),
      ("k", self.k_error_type),
      ("inh_delta", self.inh_delta_error_type),
      ("roughness", self.roughness_error_type),
      ("interface", self.interface_error_type),
    ] {
      crate::structure::ErrorType::try_from_i32(v)
        .map_err(|_| format!("group {:?}: {key} error type must be 0, 1 or 2.", self.name))?;
    }
    Ok(())
  }
}

impl BlockCfg {
  pub fn validate(&self) -> Result<(), String> {
    if self.repeat_count < 1 {
      return Err("repeat_count must be >= 1.".to_string());
    }
    crate::structure::BlockKind::try_from_i32(self.kind)
      .map_err(|e| format!("kind: {e}"))?;
    Ok(())
  }
}

'''
s = s.replace(anchor, block + anchor, 1)
io.open(p, 'w', encoding='utf-8', newline='').write(s)
print('OK')
