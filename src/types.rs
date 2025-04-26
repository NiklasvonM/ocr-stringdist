use std::collections::HashMap;

/// Type alias for a substitution key (pair of string tokens)
pub type SubstitutionKey = (String, String);

/// Type alias for a single string token key (for insertion/deletion)
pub type SingleTokenKey = String;

/// Type alias for a map of substitution costs between pairs of strings
pub type SubstitutionCostMap = HashMap<SubstitutionKey, f64>;

/// Type alias for a map of costs for single string tokens (insertion/deletion)
pub type SingleTokenCostMap = HashMap<SingleTokenKey, f64>;
