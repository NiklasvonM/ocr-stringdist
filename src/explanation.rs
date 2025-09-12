/// Represents a single operation in the edit path.
/// This is the data structure that will be returned to Python.
#[derive(Debug, Clone)]
pub enum EditOperation {
    Substitute {
        source: String,
        target: String,
        cost: f64,
    },
    Insert {
        target: String,
        cost: f64,
    },
    Delete {
        source: String,
        cost: f64,
    },
    Match {
        token: String,
    },
}

/// Represents the predecessor cell in the DP matrix, used for backtracking.
/// It stores the type of operation and the length of the tokens involved.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Predecessor {
    None, // Used for the top-left cell (0, 0)
    Substitute(usize, usize),
    Insert(usize),
    Delete(usize),
    Match(usize),
}
