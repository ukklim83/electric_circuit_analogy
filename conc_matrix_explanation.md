# What is concentration matrix?
**Simple system: two components**
- To express the fraction of material 2 over material 1 in solution, usually people use molar fraction, volume fraction, mass fraction, etc.
- Let us designate the fraction as $x$ ($x \times 100$ percent of material 2 is included in the solution).
- Then, each outlet streams can define their concentrations as $1-x : x$, which is the function of $x$
- Therefore, for the case of two components system with n outlets, the concentration of outlets can be expressed by the $n \times 1$ vector (n outlets $\times$ 1 variables)
- It can be converted with $n \times 2$ matrix if we reflect $1-x$ as a new variable. (DoF is 1, but for the simplicity, using another variable.)

**Generalization**
- Let us expand the system with m components.
- To explain the fraction of the system, $m-1$ fraction variables are needed.
- i.e. the fraction variables can be written with the following ordered pair: $(1-\sum_{i=1}^{m-1}{x_{i}}, x_{1}, x_{2}, \cdots, x_{m-1})$
- In this case, the concentration of outlets can be expressed by the $n \times m-1$ matrix (n outlets $\times$ m-1 variables)
- As well as the 2 component system, it can also be converted with $n \times m$ matrix.

**Conclusion**
- Therefore, to designate the concentration, using concentration matrix of the size $n \times m$ is reasonable.