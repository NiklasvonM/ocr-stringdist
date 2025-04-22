use std::collections::HashMap;

/// Type alias for a substitution key (pair of string tokens)
pub type SubstitutionKey = (String, String);

/// Type alias for a map of substitution costs between pairs of strings
pub type SubstitutionCostMap = HashMap<SubstitutionKey, f64>;
