// g++ -std=c++11 -Llib -I../include -I../src mxm_gen.cpp -ltaco;./a.out

#include "taco.h"
#include "taco/index_notation/transformations.h"
#include "codegen/codegen_c.h"
#include "taco/lower/lower.h"

#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <fstream>

using namespace taco; 

ir::Expr orImpl(const std::vector<ir::Expr>& v) {
  return ir::Or::make(v[0], v[1]);
}
Func OrOp("or", orImpl, {Annihilator(true), Identity(false), Commutative(), Associative()});

ir::Expr andImpl(const std::vector<ir::Expr>& v) {
  return ir::And::make(v[0], v[1]);
}
Func AndOp("and", andImpl, {Annihilator(false), Identity(true), Commutative(), Associative()});

ir::Expr addImpl(const std::vector<ir::Expr>& v) {
  return ir::Add::make(v[0], v[1]);
}
Func AddOp("add", addImpl, {Annihilator(std::numeric_limits<double>::infinity()), Identity(0), Commutative(), Associative()});

ir::Expr minImpl(const std::vector<ir::Expr>& v) {
  return ir::Min::make(v[0], v[1]);
}
Func MinOp("min", minImpl, {Identity(std::numeric_limits<double>::infinity()), Commutative(), Associative()});

ir::Expr maskImpl(const std::vector<ir::Expr>& v) {
  return v[0];
}
struct MaskAlgebra {
  IterationAlgebra operator()(const std::vector<IndexExpr>& r) {
    return Intersect(r[0], Complement(r[1]));
  }
};
Func MaskOp("mask", maskImpl, MaskAlgebra());

static bool compare(std::vector<IndexVar> vars1, std::vector<IndexVar> vars2) {
  return vars1 == vars2;
}

static IndexStmt optimizeSpGEMM(IndexStmt stmt) {
  if (!isa<Forall>(stmt)) {
    return stmt;
  }
  Forall foralli = to<Forall>(stmt);
  IndexVar i = foralli.getIndexVar();

  if (!isa<Forall>(foralli.getStmt())) {
    return stmt;
  }
  Forall forallk = to<Forall>(foralli.getStmt());
  IndexVar k = forallk.getIndexVar();

  if (!isa<Forall>(forallk.getStmt())) {
    return stmt;
  }
  Forall forallj = to<Forall>(forallk.getStmt());
  IndexVar j = forallj.getIndexVar();

  if (!isa<Assignment>(forallj.getStmt())) {
    return stmt;
  }
  Assignment assignment = to<Assignment>(forallj.getStmt());
  IndexExpr reduceOp = assignment.getOperator();

  if (!isa<Call>(assignment.getRhs())) {
    return stmt;
  }
  Call mul = to<Call>(assignment.getRhs());

  taco_iassert(isa<Access>(assignment.getLhs()));
  if (!isa<Access>(mul.getArgs()[0])) {
    return stmt;
  }
  if (!isa<Access>(mul.getArgs()[1])) {
    return stmt;
  }

  Access Aaccess = to<Access>(assignment.getLhs());
  Access Baccess = to<Access>(mul.getArgs()[0]);
  Access Caccess = to<Access>(mul.getArgs()[1]);

  if (Aaccess.getIndexVars().size() != 2 ||
      Baccess.getIndexVars().size() != 2 ||
      Caccess.getIndexVars().size() != 2) {
    return stmt;
  }

  if (!compare(Aaccess.getIndexVars(), {i,j}) ||
      !compare(Baccess.getIndexVars(), {i,k}) ||
      !compare(Caccess.getIndexVars(), {k,j})) {
    return stmt;
  }

  TensorVar A = Aaccess.getTensorVar();
  if (A.getFormat().getModeFormats()[0].getName() != "dense" ||
      A.getFormat().getModeFormats()[1].getName() != "compressed" ||
      A.getFormat().getModeOrdering()[0] != 0 ||
      A.getFormat().getModeOrdering()[1] != 1) {
    return stmt;
  }

  // I think we can to linear combination of rows as long as there are no permutations in the format and the
  // level formats are ordered. The i -> k -> j loops should iterate over the data structures without issue.
  TensorVar B = Baccess.getTensorVar();
  if (!B.getFormat().getModeFormats()[0].isOrdered() && A.getFormat().getModeFormats()[0].isOrdered() ||
      !B.getFormat().getModeFormats()[1].isOrdered() && A.getFormat().getModeFormats()[1].isOrdered() ||
      B.getFormat().getModeOrdering()[0] != 0 ||
      B.getFormat().getModeOrdering()[1] != 1) {
    return stmt;
  }

  TensorVar C = Caccess.getTensorVar();
  if (!C.getFormat().getModeFormats()[0].isOrdered()  && A.getFormat().getModeFormats()[0].isOrdered() ||
      !C.getFormat().getModeFormats()[1].isOrdered()  && A.getFormat().getModeFormats()[1].isOrdered() ||
      C.getFormat().getModeOrdering()[0] != 0 ||
      C.getFormat().getModeOrdering()[1] != 1) {
    return stmt;
  }

  // It's an SpMM statement so return an optimized SpMM statement
  TensorVar w("w",
              Type(A.getType().getDataType(), 
              {A.getType().getShape().getDimension(1)}),
              taco::dense,
              A.getFill());
  return forall(i,
                where(forall(j,
                             A(i,j) = w(j)),
                      forall(k,
                             forall(j,
                                    Assignment(w(j), mul, reduceOp)))));
}

void printToFile(std::string filename, IndexStmt stmt) {
  std::stringstream source;

  std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(source, ir::CodeGen::ImplementationGen);
  ir::Stmt compute = lower(stmt, "evaluate", true, true);
  codegen->compile(compute, true);

  std::ofstream source_file;
  source_file.open(filename + ".c");
  source_file << source.str();
  source_file.close();
}

Format UCSR({Dense, Compressed(ModeFormat::NOT_ORDERED)});
Format UZCSR({Dense, Compressed({ModeFormat::NOT_ORDERED, ModeFormat::ZEROLESS})});

int main() {
  IndexVar i("i"), j("j"), k("k");
#if 1
  Tensor<double> A("A", {200, 200}, UCSR, std::numeric_limits<double>::infinity());
  Tensor<double> B("B", {200, 200}, UCSR, std::numeric_limits<double>::infinity());
  Tensor<double> C("C", {200, 200}, UCSR, std::numeric_limits<double>::infinity());
  A(i,j) = Reduction(MinOp(), k, AddOp(B(i,k), C(k,j)));
#else
  Tensor<bool> A("A", {200, 200}, UCSR);
  Tensor<bool> B("B", {200, 200}, UZCSR);
  Tensor<bool> C("C", {200, 200}, UZCSR);
  A(i,j) = Reduction(OrOp(), k, AndOp(B(i,k), C(k,j)));
#endif
  IndexStmt stmt = A.getAssignment().concretize();
  stmt = reorderLoopsTopologically(stmt);
  stmt = optimizeSpGEMM(stmt);
  stmt = stmt.assemble(A.getTensorVar(), AssembleStrategy::Insert);
  IndexVar qi = to<Forall>(to<Assemble>(stmt).getQueries()).getIndexVar();
  stmt = stmt.parallelize(i, ParallelUnit::CPUThread,
                          OutputRaceStrategy::NoRaces)
             .parallelize(qi, ParallelUnit::CPUThread,
                          OutputRaceStrategy::NoRaces);
  stmt = scalarPromote(stmt);
  printToFile("mxm", stmt);
  return 0;
}
