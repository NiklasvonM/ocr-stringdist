mod longest_tokens;
mod weighted_levenshtein;

pub use longest_tokens::longest_key_string_length;
pub use weighted_levenshtein::{custom_levenshtein_distance_with_cost_map, OcrCostMap};

#[cfg(feature = "python")]
mod rust_stringdist;
#[cfg(feature = "python")]
pub use rust_stringdist::_rust_stringdist;
