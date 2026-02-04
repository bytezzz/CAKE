use egg::*;

define_language! {
    enum ArrowQuiverLanguage {
        "read-parquet" = ReadParquet([Id ; 3]),
        "cols" = ColList(Vec<Id>),

        // sel quiver -> quiver
        "filter" = Filter([Id ; 2]),

        // const col -> sel
        "ltc" = LessThanConst([Id ; 2]),

        // const col sel -> sel
        "partial-ltc" = PartialLessThanConst([Id ; 3]),

        "and" = And([Id ; 2]),
        "or" = Or([Id ; 2]),
        "not" = Not(Id),
        "is-null" = IsNull(Id),
        "is-not-null" = IsNotNull(Id),
        "hint-vec" = HintSelectionVector(Id),
        "hint-bitmap" = HintBitmap(Id),
        "true" = Id,
        IntegerConst(i64),
        ColName(Symbol),
    }
}

type EGraph = egg::EGraph<ArrowQuiverLanguage, ()>;

fn make_rules() -> Vec<Rewrite<ArrowQuiverLanguage, ()>> {
    let one_way = vec![
        rewrite!("elim-double-negate"; "(not (not ?a))" => "?a"),
        rewrite!("elim-null-negate"; "(not (is-null ?a))" => "(is-not-null ?a)"),
        rewrite!("elim-not-null-negate"; "(not (is-not-null ?a))" => "(is-null ?a)"),
        rewrite!("hint-vec"; "(partial-ltc ?a ?b ?c)" => "(partial-ltc ?a ?b (hint-vec ?c))"),
        rewrite!("hint-bitmap"; "(partial-ltc ?a ?b ?c)" => "(hint-bitmap (partial-ltc ?a ?b ?c))"),
        rewrite!(
            "elim-redundant-ltc"; 
            "(partial-ltc ?a ?b (partial-ltc ?c ?b ?d))" => "(partial-ltc ?c ?b ?d)"
            if is_gt_eq("?a", "?c")),
    ];
    let two_way = vec![
        rewrite!("combine-split-filters"; "(filter ?a (filter ?b ?c))" <=> "(filter (and ?a ?b) ?c)"),
        rewrite!("commute-and"; "(and ?a ?b)" <=> "(and ?b ?a)"),
        rewrite!("commute-or"; "(or ?a ?b)" <=> "(and ?b ?a)"),
        rewrite!("pull-not-over-and"; "(and (not ?a) (not ?b))" <=> "(not (or ?a ?b))"),
        rewrite!("pull-not-over-or"; "(or (not ?a) (not ?b))" <=> "(not (and ?a ?b))"),
        rewrite!("ltc-to-partial-ltc"; "(and (ltc ?b ?c) ?a)" <=> "(partial-ltc ?b ?c ?a)"),
        rewrite!("pull-partial-ltc"; "(and (partial-ltc ?a ?b ?c) ?d)" <=> "(partial-ltc ?a ?b (and ?c ?d))")
    ].concat();
    [one_way, two_way].concat()
}

fn is_gt_eq(v1: &'static str, v2: &'static str) -> impl Fn(&mut EGraph, Id, &Subst) -> bool {
    let v1 = v1.parse().unwrap();
    let v2 = v2.parse().unwrap();
    move |egraph, _id, subst| {
        let v1 = &egraph[subst[v1]].nodes.iter().find_map(|el| match el {
            ArrowQuiverLanguage::IntegerConst(x) => Some(x),
            _ => None,
        });

        let v2 = &egraph[subst[v2]].nodes.iter().find_map(|el| match el {
            ArrowQuiverLanguage::IntegerConst(x) => Some(x),
            _ => None,
        });

        match (v1, v2) {
            (Some(v1), Some(v2)) => v1 >= v2,
            _ => false,
        }
    }
}

struct SimpleWeightedCostFn;
impl CostFunction<ArrowQuiverLanguage> for SimpleWeightedCostFn {
    type Cost = usize;

    fn cost<C>(&mut self, enode: &ArrowQuiverLanguage, mut costs: C) -> Self::Cost
    where
        C: FnMut(Id) -> Self::Cost,
    {
        let node_cost = match enode {
            ArrowQuiverLanguage::ReadParquet(_) => 100,
            ArrowQuiverLanguage::ColList(_) => 0,
            ArrowQuiverLanguage::Filter(_) => 50,
            ArrowQuiverLanguage::LessThanConst(_) => 2,
            ArrowQuiverLanguage::PartialLessThanConst(_) => 5,
            ArrowQuiverLanguage::And(_) => 1,
            ArrowQuiverLanguage::Or(_) => 1,
            ArrowQuiverLanguage::Not(_) => 1,
            ArrowQuiverLanguage::IsNull(_) => 1,
            ArrowQuiverLanguage::IsNotNull(_) => 1,
            ArrowQuiverLanguage::HintSelectionVector(_) => 3,
            ArrowQuiverLanguage::HintBitmap(_) => 3,
            ArrowQuiverLanguage::Id => 0,
            ArrowQuiverLanguage::IntegerConst(_) => 0,
            ArrowQuiverLanguage::ColName(_) => 0,
        };

        enode.fold(node_cost, |sum, node| sum + costs(node))
    }
}

fn optimize(p: &str) -> String {
    let expr: RecExpr<ArrowQuiverLanguage> = p.parse().unwrap();
    let runner = Runner::default().with_expr(&expr).run(&make_rules());
    let root = runner.roots[0];
    let extractor = Extractor::new(&runner.egraph, SimpleWeightedCostFn);
    let (best_cost, best) = extractor.find_best(root);
    println!("{} to {} with cost {}", expr, best, best_cost);

    best.to_string()
}

#[cfg(test)]
mod tests {
    use super::optimize;

    #[test]
    fn test_simple_pred() {
        optimize(
            "(filter 
  (and 
    (not (not (is-not-null #0))) 
    (and (ltc 5 #1) 
         (and (is-not-null #1) (ltc 3 #1))
    )
  )
  (read-parquet x y z)
)",
        );
    }

    #[test]
    fn test_simple_read() {}
}
