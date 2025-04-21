use std::collections::HashMap;

/// Calculates the length of the longest string found within the key tuples of a HashMap.
pub fn longest_key_string_length<V>(map: &HashMap<(String, String), V>) -> usize {
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
        let mut map = HashMap::new();
        map.insert(("apple".to_string(), "banana".to_string()), 1); // 5, 6
        map.insert(("kiwi".to_string(), "grapefruit".to_string()), 2); // 4, 10
        map.insert(("short".to_string(), "tiny".to_string()), 3); // 5, 4

        assert_eq!(longest_key_string_length(&map), 10); // "grapefruit"
    }

    #[test]
    fn test_longest_key_string_length_first_element() {
        let mut map = HashMap::new();
        map.insert(("a_very_long_string".to_string(), "short".to_string()), 1); // 18, 5
        map.insert(("medium".to_string(), "small".to_string()), 2); // 6, 5

        assert_eq!(longest_key_string_length(&map), 18);
    }

    #[test]
    fn test_longest_key_string_length_empty_map() {
        let map: HashMap<(String, String), bool> = HashMap::new();
        assert_eq!(longest_key_string_length(&map), 1);
    }

    #[test]
    fn test_longest_key_string_length_empty_strings() {
        let mut map = HashMap::new();
        map.insert(("".to_string(), "".to_string()), 1);
        map.insert(("a".to_string(), "".to_string()), 2);

        assert_eq!(longest_key_string_length(&map), 1);
    }
}
