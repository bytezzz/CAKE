use crate::{
    predicate::Predicate, selection::sel_intersect::Intersector, ArrowQuiver, Operator, Transform,
};

pub enum AQExpr {
    Input,
    Filter(Predicate, Box<AQExpr>),
    Materialize(Box<AQExpr>),
}

impl Operator for AQExpr {
    fn name(&self) -> String {
        "Expression".to_string()
    }
}

impl Transform for AQExpr {
    fn transform(&self, ctx: &crate::AQExecutorContext, quiver: ArrowQuiver) -> Vec<ArrowQuiver> {
        vec![self.apply(quiver, &ctx.intersector)]
    }
}

impl AQExpr {
    pub fn apply(&self, aq: ArrowQuiver, intersector: &Intersector) -> ArrowQuiver {
        match self {
            AQExpr::Input => aq,
            AQExpr::Filter(predicate, c) => {
                let aq = c.apply(aq, intersector);
                let sel = predicate.apply(&aq, intersector);
                aq.with_selection(sel)
            }
            AQExpr::Materialize(c) => {
                let aq = c.apply(aq, intersector);
                aq.materialize()
            }
        }
    }
}
