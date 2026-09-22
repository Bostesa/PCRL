"""Apply Terminal 3's text repairs to Terminal 2's pinned manuscript.

Input: main.tex and references.bib as committed at
30a6fd19e17453bc8c421a65491b5b482ab9291f (papers/pcrl_satml_final_v1/). Every
replacement must match exactly once, or the script fails. The registered title
and abstract are NOT modified here; the optional abstract edits are listed in
MANUSCRIPT_PATCH.md.

Usage: python build_manuscript_patch.py <in_dir> <out_dir> <bib_additions>
"""
import sys
from pathlib import Path

EDITS = [
    # P-C2: contribution item 2 -- the identity is not original (M7)
    ("item2",
     r"""\item \textbf{An auditing result that survives checking} (\S\ref{sec:audit}): a prevalence-weighted
aggregate one-hot regression score can conceal a much larger sensitive direction, which a class-aware
dominant-axis audit exposes; we give the exact convex-combination identity behind it, and the precise
limits of the linear guarantees it is often paired with.""",
     r"""\item \textbf{An audit caution that survives checking} (\S\ref{sec:audit}): the pooled one-hot
regression score is a prevalence-weighted average of per-class scores, so it can conceal a much larger
rare-class direction that a per-class (dominant-axis) audit exposes; we state the precise limits of the
linear guarantees it is often paired with, including a tight bound for approximate composition."""),
    # P-C3: contribution item 3 -- an instance of an established programme
    ("item3",
     r"""\item \textbf{A task-directed finite release} (\S\ref{sec:method}) optimised under explicit
conditional-disclosure constraints, with its optimisation assumptions, its relationship to established
convex programmes, and an exact fixture separating what it can and cannot do.""",
     r"""\item \textbf{A task-directed finite release} (\S\ref{sec:method}): an instance of an established
convex leakage--distortion programme for one recipient, with its optimisation assumptions, the coarse
conditioning it actually uses, and an exact fixture separating what it can and cannot do."""),
    # E4: measurement semantics
    ("measure",
     r"""\paragraph{What a number means here} Every recovery figure is the log-loss reduction an independently
fitted, validation-selected attacker achieves relative to the same family restricted to $H$. It is a
floor on what is recoverable, never a ceiling, and it is not an estimate of conditional mutual
information~\cite{mcallester2020formal}: a restricted predictor can improve after receiving a deterministic feature $Z=f(H)$ even
where $I(S;Z\mid H)=0$, and a failed attack bounds nothing from above.""",
     r"""\paragraph{What a number means here} Every recovery figure is the log-loss reduction an independently
fitted, validation-selected attacker achieves relative to the same family restricted to $H$. Absolute
recovery---the entropy baseline minus the held-out loss of an attacker fixed before evaluation---is, up
to sampling error, a lower bound on the view's mutual information with the attribute. The increment over
the $H$-restricted fit is a difference of two such lower bounds and bounds $I(S;Z\mid H)$ in neither
direction~\cite{mcallester2020formal}: a restricted predictor can improve after receiving a deterministic
feature $Z=f(H)$ even where $I(S;Z\mid H)=0$, and a failed attack bounds nothing from above."""),
    # E7a / M7: dominant-axis definition
    ("da_def",
     r"""so a rare class contributes almost nothing to the number a compliance report prints. Auditing the
\emph{dominant axis} instead---the worst single direction across classes---exposes what the aggregate
smooths away.""",
     r"""so a rare class contributes almost nothing to the number a compliance report prints. This is the
familiar variance-weighted multi-output $R^2$, not a new identity; it holds exactly when both scores share
rows, centring and ridge penalty. Auditing the \emph{dominant axis} instead---the largest per-class
one-versus-rest score, itself only a lower bound on the best linear sensitive direction---exposes what the
aggregate smooths away."""),
    # E7b/c: identity "validation" and rarest-class statement
    ("da_val",
     r"""one configuration the aggregate reads $0.027$ where the dominant axis reads $0.288$. The identity itself
was validated on every multi-class pair-seed, maximum absolute residual $0.0021$.""",
     r"""one configuration the aggregate reads $0.027$ where the dominant axis reads $0.288$, on the rarest
class (prevalence $0.9\%$), for which the aggregate alone cannot bound the class score. Replaying the
stored per-class scores reproduces the stored aggregate within $2\times10^{-5}$ in $31$ of $33$ cells; the
two exceptions trace to a single-precision aggregate in the historical evaluation code."""),
    # Table I caption: in-sample
    ("da_cap",
     r"""pair-seeds of the canonical checkpoints. The aggregate understates the recoverable direction, most on the
imbalanced attribute.""",
     r"""pair-seeds of the canonical checkpoints (in-sample ridge scores on held-out test representations). The
aggregate understates the recoverable direction, most on the imbalanced attribute."""),
    # E6 + M6: API and the convex-loss statement
    ("api",
     r"""The claim is withdrawn and the corresponding API now refuses to run. The valid statement is narrower and
about squared loss only: zero cross-covariance means affine least-squares prediction cannot improve on the
optimal constant predictor on that distribution.""",
     r"""The claim is withdrawn and the corresponding API refuses to run in the evaluated code. The valid
statement concerns losses convex in an affine prediction: zero cross-covariance means no affine predictor
improves on the best constant under squared, logistic or any other such loss on that
distribution~\cite{belrose2023leace}; thresholded accuracy is not such a loss."""),
    # T3: composition with the lambda_min lemma
    ("compose",
     r"""Second, \textbf{composition is exact only at exact zero}. Two views with exactly zero cross-covariance
with the same signal retain exactly zero under concatenation. Approximate leakage does not inherit this:
$h_1=N+\delta S$ and $h_2=N-\delta S$ each carry $R^2=\delta^2/(1+\delta^2)$ and their difference recovers
$S$. Neither statement supplies independence or any guarantee against a nonlinear attack.""",
     r"""Second, \textbf{composition is exact only at exact zero}. Two views with exactly zero cross-covariance
with the same signal retain exactly zero under concatenation, yet can determine it nonlinearly: with
independent signs $V,S$, the views $SV$ and $V$ each have zero covariance with $S$ and their product is
$S$. Approximate leakage composes with a conditioning penalty: for views with nonsingular covariances,
$R^2(h_1,\dots,h_k)\le\sum_p R^2(h_p)/\lambda_{\min}(R)$, where $R$ is the block-whitened cross-view
correlation (a Rayleigh--Ritz bound). It is tight: $h_1=N+\delta S$ and $h_2=N-\delta S$ each carry
$R^2=\delta^2/(1+\delta^2)$, $\lambda_{\min}(R)=2\delta^2/(1+\delta^2)$, and their difference recovers $S$."""),
    # M3 + Creager
    ("ufate",
     r"""al.~\cite{sadeghi2019sarl,sadeghi2021optnetarl,sadeghi2022ktopt} and the conditional extension
U-FaTE~\cite{dehdashtian2024ufate} solve the same shape of problem when the representation is the
releaser's to change.""",
     r"""al.~\cite{sadeghi2019sarl,sadeghi2021optnetarl,sadeghi2022ktopt} and U-FaTE~\cite{dehdashtian2024ufate},
which conditions its dependence measure on the target label, solve the same shape of problem when the
representation is the releaser's to change. Flexibly fair representations let each task drop a different
subset of sensitive attributes from one shared encoder~\cite{creager2019ffvae}."""),
    # M1, M2, M4, M5: finite-alphabet paragraph
    ("finite",
     r"""\paragraph{Finite-alphabet privacy--utility optimisation} The convex programme of \S\ref{sec:method} is
established. Rassouli and G\"und\"uz~\cite{rassouli2021perfect} give the perfect-privacy characterisation,
the nullspace feasibility criterion and the linear-programme form of the zero-budget linear-cost problem;
the privacy funnel of Makhdoumi et al.~\cite{makhdoumi2014funnel} is nonconvex precisely because of a
utility constraint we do not impose. Multi-user and side-information formulations of utility--privacy
tradeoffs~\cite{sankar2013utility,liao2019sideinfo} design a sanitised release for many consumers. We
claim no new optimisation result: the nullspace criterion and the convexity facts are theirs, and what is
ours is the release contract, the per-recipient and coalition constraint structure, and an evaluated
operating point.""",
     r"""\paragraph{Finite-alphabet privacy--utility optimisation} The convex programme of \S\ref{sec:method} is
established: finite-alphabet leakage--distortion design is convex for a fixed joint
law~\cite{calmon2012privacy}, including after quantising to a finite code~\cite{salamatian2015managing}.
Rassouli and G\"und\"uz~\cite{rassouli2021perfect} characterise when perfect privacy is feasible through a
nullspace condition and reduce the zero-leakage problem with mutual-information utility to a linear
programme; our zero-budget case is conditional and is linear by the same argument. The privacy funnel of
Makhdoumi et al.~\cite{makhdoumi2014funnel} is nonconvex precisely because of a utility constraint we do
not impose. A new release has been designed beside fixed earlier releases under an incremental-leakage
budget~\cite{erdogdu2015continual}, and per-party and collusion mutual-information constraints on
sequential releases were studied recently~\cite{taylor2026collusion}; since $C$ is fixed,
$I(S;Z\mid C)=I(S;Z,C)-I(S;C)$, so our conditional budgets are joint-view budgets with a shifted
constant. Side information at the recipient appears in~\cite{sankar2013utility,liao2019sideinfo}, and
randomised finite pre-processing under fairness constraints in~\cite{calmon2017optimized}. We claim no new
optimisation result. What is ours is the setting---the conditioning view is a third party's published
prediction service that a coalition partner also holds---the measurement contract, and an evaluated
operating point."""),
    # E1: label-free code
    ("construct",
     r"""\paragraph{Construction} A label-free code $T=g(X_A)$ over permitted $A$-side inputs is mapped to a finite
output alphabet by a row-stochastic channel $Q$, and the recipient receives one sampled token.""",
     r"""\paragraph{Construction} Two residence-supervised teachers, $p(X_A,H_A)$ and $b(H_A)$, define a residual
logit $r=\operatorname{logit}p-\operatorname{logit}b$; its $32$ quantile cells form a code $T=g(X_A,H_A)$,
into which no protected label enters at runtime. $T$ is mapped to a finite output alphabet by a
row-stochastic channel $Q$, and the recipient receives one sampled token."""),
    # E2, E8: local vs coalition, solver status
    ("convex",
     r"""with a cost table $D$ fixed before optimisation. Under the mechanism's Markov structure $Z-T-(S,C)$ and a
fixed fitted law, each conditional mutual information is affine in $Q$ inside a jointly convex relative
entropy, so the feasible set is convex and the objective linear.""",
     r"""with a cost table $D$ fixed before optimisation. Under the mechanism's Markov structure $Z-T-(S,C)$ and a
fixed fitted law, each conditional mutual information is affine in $Q$ inside a jointly convex relative
entropy, so the feasible set is convex and the objective linear. A \emph{local} policy imposes the $r=A$
constraints only; a \emph{coalition} policy adds $r=AB$. The selected utility release is local, so its
coalition view is audited but not constrained. We report solver status and replay primal feasibility;
no duality gap was recorded, so optimality of the fitted programme is solver-reported, not certified."""),
    # M2 in the attribution paragraph
    ("attrib",
     r"""contract, not a new optimisation result. The finite-alphabet privacy--utility problem, the nullspace
feasibility criterion and the linear-programme form at zero budget are prior work; the privacy funnel is
nonconvex precisely because of a utility constraint we do not impose.""",
     r"""contract, not a new optimisation result. The convex finite-alphabet leakage--distortion
problem~\cite{calmon2012privacy}, releases beside fixed earlier releases~\cite{erdogdu2015continual},
collusion constraints~\cite{taylor2026collusion} and the nullspace feasibility
criterion~\cite{rassouli2021perfect} are prior work; the privacy funnel is nonconvex precisely because of
a utility constraint we do not impose."""),
    # E3: cell counts
    ("cells",
     r"""\emph{fitted finite model} whose conditioning view is a coarsened partition. It is not a population""",
     r"""\emph{fitted finite model} whose conditioning view is a coarsened partition---two cells of $H_A$ for
$A$, four of $(H_A,H_B)$ for $AB$. It is not a population"""),
    # E1 in the baselines paragraph
    ("baselines",
     r"""releases} ($D_{17}$, $D_{33}$) use the same label-free code and action""",
     r"""releases} ($D_{17}$, $D_{33}$) use the same residence-supervised code and action"""),
    # E5: backbone in the lineage section
    ("lineage",
     r"""This work descends from a purpose-conditioned representation line in which one shared frozen backbone
carried per-purpose adapters,""",
     r"""This work descends from a purpose-conditioned representation line in which one shared frozen, randomly
initialised backbone carried per-purpose adapters,"""),
    # corrections list: name the skipped certificate, add three items
    ("corr4",
     r"""\item \textbf{A proposed near-optimality certificate is not an implemented result.} It was explicitly
skipped for want of the necessary numerical check and is not claimed here.""",
     r"""\item \textbf{A proposed Sadeghi--Boddeti near-optimality certificate is not an implemented result.}
It was explicitly skipped for want of the necessary numerical check and is not claimed here.
\item \textbf{The encoder line's backbone was not pretrained.} It was described as pretrained on the
union of permitted tasks; in the implementation that produced its results it was a seeded random MLP,
frozen at initialisation.
\item \textbf{A rank floor for single-layer edits is not a floor for the trained adapters.} Erasure below
$\sum_{i>r}\sigma_i^2/\operatorname{tr}\operatorname{Cov}(A)$ is impossible for one rank-$r$
multiplicative edit of a fixed representation, but the trained adapters edit every layer with
nonlinearities in between; a rank-one edit of the first layer can remove a rank-two sensitive
cross-covariance entirely."""),
]


def main(src, dst, bib_add):
    src, dst = Path(src), Path(dst)
    tex = (src / "main.tex").read_text()
    for name, old, new in EDITS:
        n = tex.count(old)
        if n != 1:
            raise SystemExit(f"edit {name}: expected 1 match, found {n}")
        tex = tex.replace(old, new)
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "main.tex").write_text(tex)
    bib = (src / "references.bib").read_text().rstrip("\n") + "\n\n" + Path(bib_add).read_text()
    (dst / "references.bib").write_text(bib)
    print(f"applied {len(EDITS)} edits")


if __name__ == "__main__":
    main(*sys.argv[1:4])
