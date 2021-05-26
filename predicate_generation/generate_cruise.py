import sys
from typing import List
from sympy import symbols, solve, simplify
from sympy.physics.units import length, velocity, acceleration, time
from sympy.physics.units.systems.si import dimsys_SI

LEVEL = 1
UES_DIFF = True

def printVals(vals):
  cnt = 0
  for genVar in vals.keys():
    cnt += len(vals[genVar])
    print("\n#####", genVar, "#######")
    for eq in vals[genVar]:
      print(f"{genVar} = {eq}")
  print("# equations:", cnt)

baseVals = [vE, vF, vMin, vMax, d, aMax, aMin, dSafe, tOne] = symbols("vE,vF,vMin,vMax,d,aMax,aMin,dSafe,tOne")
genVars = [a, v, s, t] = symbols("a,v,s,t")

units = {vE: velocity, vF: velocity, vMin: velocity, vMax: velocity, d: length,
  aMax: acceleration, aMin: acceleration, dSafe: length, tOne: time,
  a: acceleration, v: velocity, s: length, t:time}

baseEqs = [
  a*t**2/2 + v*t - s,
  a*t - v,
]

concat = lambda l : [x for lst in l for x in lst]

baseIdentities = {
  var: concat([solve(eq, var) for eq in baseEqs if var in eq.free_symbols])
  for var in set().union(*[eq.free_symbols for eq in baseEqs])
}
#print("base identities:")
#printVals(baseIdentities)
'''
In the end the propositions will have the form f(x) '<=' 0.
So, choose one generic variable, solve for it, and plug in some value for the others.
'''
vals = {
  var: [val for val in baseVals if units[val] == units[var]]
  for var in baseIdentities.keys()
}

def tryAllSubsts(res:List, eq):
  freeVars = eq.free_symbols - set(baseVals)
  if len(freeVars) == 0:
    res.append(eq)
    return
  freeVar = next(iter(freeVars))
   # plug value
  for val in vals[freeVar]:
    newEq = eq.subs(freeVar, val)
    tryAllSubsts(res, newEq)

for i in range(LEVEL):
  vals2 = {var:[] for var in baseIdentities.keys()}
  if UES_DIFF:
    for genVar in baseIdentities.keys():
      valSet = set()
      for v1 in vals[genVar]:
        valSet.add(v1)
        for v2 in vals[genVar]:
          tSum = v1 + v2
          tDif = v1 - v2
          valSet.add(tSum)
          if v1 != v2:
            valSet.add(tDif)
      vals[genVar] = list(valSet)

  for genVar in baseIdentities.keys():
    for eq in baseIdentities[genVar]:
      newVals = []
      tryAllSubsts(newVals, eq)
      vals2[genVar] += newVals
    vals2[genVar] = list(set(vals2[genVar]))
  vals = vals2

print(f"found {list(map(len, vals.values()))}", file=sys.stderr)
print(f"found {sum(map(len, vals.values()))}", file=sys.stderr)
#printVals(vals)
# make unique

# now convert to dtcontrol format

dtMap = {
  vE: "x_3",
  vF: "x_5",
  vMin: "-6",
  vMax: "20",
  d: "x_2",
  aMax: "2",
  aMin: "-2",
  dSafe: "5",
  tOne: "1",
}
subtitutions = [(k, v) for k, v in dtMap.items()]
terms = set()
def printDt():
  for genVar in vals.keys():
    #print("\n#####", genVar, "#######")
    for eq in vals[genVar]:
      #expr = simplify(eq.subs(subtitutions))
      expr = eq.subs(subtitutions)
      if len(expr.free_symbols) > 0:
        term = f"{expr} <= c_1"
        if term not in terms and not "oo" in term: # skip infinity 
          terms.add(term)
          print(term)



printDt()
