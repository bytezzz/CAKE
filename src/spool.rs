use std::sync::Mutex;

use crate::{ArrowQuiver, Operator, Sink};

pub struct Spool {
    pub data: Mutex<Vec<ArrowQuiver>>,
}

impl Default for Spool {
    fn default() -> Self {
        Self::new()
    }
}

impl Spool {
    pub fn new() -> Self {
        Spool {
            data: Mutex::new(vec![]),
        }
    }
}

impl Operator for Spool {
    fn name(&self) -> String {
        "Spool".to_string()
    }
}

impl Sink for Spool {
    type Output = Vec<ArrowQuiver>;

    fn sink(&self, _ctx: &crate::AQExecutorContext, quiver: ArrowQuiver) {
        if quiver.num_rows() == 0 || quiver.sel().is_none_valid() {
            return;
        }
        self.data.lock().unwrap().push(quiver);
    }

    fn finish(self: Box<Self>, _ctx: &crate::AQExecutorContext) -> Self::Output {
        self.data.into_inner().unwrap()
    }
}
