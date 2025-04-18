mod weighted_levenshtein;

pub use weighted_levenshtein::{
    custom_levenshtein_distance, custom_levenshtein_distance_with_cost_map,
    vec_custom_levenshtein_distance, vec_custom_levenshtein_distance_with_cost_map, OcrCostMap,
};

#[cfg(feature = "python")]
mod rust_stringdist;
#[cfg(feature = "python")]
pub use rust_stringdist::_rust_stringdist;
