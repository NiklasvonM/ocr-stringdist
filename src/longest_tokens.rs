use crate::types::SubstitutionCostMap;

/// Calculates the length of the longest string found within the key tuples of a HashMap.
pub fn longest_key_string_length(map: &SubstitutionCostMap) -> usize {
    map.keys()
        .flat_map(|(s1, s2)| [s1.len(), s2.len()].into_iter())
        .max()
        .unwrap_or(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_longest_key_string_length_basic() {
        let mut map = SubstitutionCostMap::new();
        map.insert(("apple".to_string(), "banana".to_string()), 1.0); // 5, 6
        map.insert(("kiwi".to_string(), "grapefruit".to_string()), 2.0); // 4, 10
        map.insert(("short".to_string(), "tiny".to_string()), 3.0); // 5, 4

        assert_eq!(longest_key_string_length(&map), 10); // "grapefruit"
    }

    #[test]
    fn test_longest_key_string_length_first_element() {
        let mut map = SubstitutionCostMap::new();
        map.insert(("a_very_long_string".to_string(), "short".to_string()), 1.0); // 18, 5
        map.insert(("medium".to_string(), "small".to_string()), 2.0); // 6, 5

        assert_eq!(longest_key_string_length(&map), 18);
    }

    #[test]
    fn test_longest_key_string_length_empty_map() {
        let map = SubstitutionCostMap::new();
        assert_eq!(longest_key_string_length(&map), 1);
    }

    #[test]
    fn test_longest_key_string_length_empty_strings() {
        let mut map = SubstitutionCostMap::new();
        map.insert(("".to_string(), "".to_string()), 1.0);
        map.insert(("a".to_string(), "".to_string()), 2.0);

        assert_eq!(longest_key_string_length(&map), 1);
    }
}
