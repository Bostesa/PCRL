# Task-conditioned continuous representation study

This is an exploratory method search designed from prior outcomes on repeatedly used Adult people. New fits, methods, locks and model seeds do not create fresh confirmation.

## Purpose and interfaces

Test an explicit pair conditional kernel-power penalty in trainable recipient encoders beyond published local conditional protection. Two separate83→64→64→16 ReLU encoders with algebraically folded useful scaler/logistic source heads; encoderfloat32, ownheadfloat64.21152encoderparameters+136headparameters=21288. Each recipient gets actual z, centred logits, probabilities and first-index-argmax decision:21/29dimensions, alignedpair50. Scores-only3/7/10dimensions are a separate service and secondary family. No clean teacher co-release, labels, IDs or flags. Decisions may change.

## Source and roles

Latest execution ecf01fe177c4d9a806ddb7d0cec02ca46e630aa6; publication8382c0df6550a84565fe2c2177fb547e5a5915a2. Exact U0 deployed scaler/LogisticRegression heads, not old affine heads; source fold parity≤2e-12 and exact decisions, independent source replay passed. Ucal fixed historical global-temperature law, no new calibration. All pretrained U0 parameters saw all FIT, historical head selection used HEAD outcomes and Ucal family previously used INNER; internal FIT selection is not newly independent. No barred labels/arrays read.

Admitted FIT15434/15428groups; ATT4064/4061; INNER2235/2234. FITfold0+1:10223rows; fold2:5211rows. Updates use folds0/1; selection fold2 only. INNERtask labels open only after ALLfamily nominees/commonconfiguration/diagnosticfallbacks frozen. INNERSEX opens only after ALLcandidate releases, readerbanks and four independent refits frozen. All scientific stage pins/guards are checked before access.

## Objectives and normalization

Official code280a178abc86802cef58f99ab3eff4d2365ddacb MMD-B-Fair core retained literally: offdiagonal MMD U-statistic, official biased rowmean variance proxy (missing paper/Bfactor explicitly disclosed), variancefloor1e-6 and Gaussian test-power CDF.32 distinctgroups/SEX per supported stratum, sampled withoutreplacement, hypothetical4blocks of8. Strongest shared relative bandwidth across weighted strata (max outside weighted sum). Target power coefficient−1perrecipient plus population-weighted CE. Multiclass target uses frozen frequency-weighted unordered class contrasts. Pair kernel product of actual recipient z+centred-own-head-logit kernels; no permutation. Local and pair conditioning use true task/pair labels during FIT only. Unsupported mass not renormalized; task training and evaluation include all people.

Detached diagonal normalization from fixed≤256 trainingreference rows refreshed everyepoch; initial-scale-relative floors max(1e-12,1e-8initialstd), neverdropcoords. Reference distance strict-upper-triangle lower median normalized squared distance; bank(.25,.5,1,2,4)^2. Normalization is a fitting transform, not an eraser. Log variances/floors/headnorms/distances and task/sensitive/target gradients.

## Matched families and recipe

U0zero-update reference, U_NEW CEcontinuation, U_TASKPOWER CE−targets, JOINT_CONDITIONAL locals+pair, LOCAL_TWIN exactsamejointlocalcoefficients removingpair, LOCAL_CONDITIONAL separately calibrated strongerlocal, JOINT_UNCONDITIONAL same nominaljointcoefficients without taskconditioning, RAW_JOINT validatedordinaryonlineadversarial, SEQ12/SEQ21 freeze-first conditional orders. No optional family.

Adam1e-4,betas(.9,.999),eps1e-8,wd0,clip5,batch256;20epochs save0/5/10/20. Separate uniformtaskminibatches; privacybalancing does not alter task priors. Taskweights1each. Forty taskupdates/epoch,800updates/encoder/head at20. Same seeded task permutations across comparable arms. Nominal strength ladder(.3,1,3) units of16fixedinitialU0 taskCE/sensitive-only gradient norm draws. All encoder/head parameters, squared-norm sums float64; sixteen-norm arithmetic median. Joint normalization sharedexactlytwin/unconditional/SEQ; localstrong andRAW separately. Require finite medians>1e-12 and largest3×coefficient≤1e4; failednormalization disables affectedfits, no silentreplacement. Compute constants only after SCIENCE lock.

RAWcritics linear/MLP32/MLP64 eachrecipient/pair, Adam3e-3,80warmupdates/critic sourcee0 sharedallstrengths; fiveupdates/critic/modelstep,9critics. Strongest CE risk1−min(CE,Htrainingprior)/Htrainingprior. No target-power term. RAWfit/calibration steps separately charged.

Sequentialfirst stage only firstlocal+target; freezeencoderANDhead, secondstage secondlocal+pair+target actualfrozenfirst. First20prefixes0/5/10/20; secondbranches5/10/20 give e/e candidateupdates. Same4checkpoints,55 aggregate encoderepochs vs40 joint,37.5%extra searchcompute favouringcontrolcharged/disclosed. Everysequentialnominee including0 ranked with BOTHlocal+pair fixed-selection diagnostic; frozenfirstconstant cannotbedropped acrossprefixes.

## FIT-only nomination and gates

One common strength+epoch forbothseeds per family. Compare FITfold2 tasks to exactU0e0 ONLY using scales acc.010,NLL.010,Brier.005: a training surrogate, not direct U0/Ucal FITgate. Protected nominee minimizes family sensitive diagnostic among commonconfigs whose worstnormalizedtaskdeviation≤1; thenworstdeviation,totalNLL,strengthorder,epochorder. Utility selects min totalNLL among≤1. Fallback alwaysminworstdeviation thenNLL/frozenorder, chosen/hashfrozen BEFOREINNER. ExactLOCAL_TWIN at selectedJOINT strengthANDepoch. Ifnone surrogatequalified usefallback as fixednominee. Actual INNER pointgate requires all24original clauses against BOTHU0/Ucal. Failednominee triggersonly its alreadyfrozen fallback for descriptiveattacks; neverpromote that fallback or reselectstrength after INNER. Source aliases explicitlyproved/reported.

## Attacks and inference

Fresh ATTgroup-disjoint train/validation; fixed sourcehashsplit. Modelbase2026101205+seed; ATTnull2026101206; INNERnull2026101207; bootstrap2026101208. Nine-reader slate perview: full LR/MLP/HGB/defense-awareMLP, representation LR/MLP/HGB, output LR/HGB. Ownfinite-variance standardization keeps small finitecoordinates; orientation onlyATTvalidation. Pair envelope includes all ignore-other readers. Shuffled/null/planted controls mandatory. All transmittedscalars included. FIT Laplace1 true-label-only table and conditional/oracle diagnostics separate, neverprimary inferenceinputs/targets.

2048pairedduplicategroupmultinomial resamples, ascendinguniques, sharedallarms/seeds. Readerenvelope recomputed perreplicate then equalpairedseedmean. Fixedmodels/readers; no stochasticrelease drawvariance/df2. PrimaryK51:JOINT24utility plus9matchedcomparators×3meanprivacycontrasts. ComparatorpoolU0,U_NEW,U_TASKPOWER,LOCAL_TWIN,LOCAL_CONDITIONAL,JOINT_UNCONDITIONAL,RAW_JOINT,SEQ12,SEQ21. UtilitysecondaryK240(all10arms×24); separate scoreprivacyK33(9newcomparators+oldP0/S12×3). Keepmissingdependencies inK. Approximate conditional-development Bonferroni t critical1−.05/(2K),df2233 times pairedbootstrapSDddof1. Descriptive perseed/mean absoluteAUC95CIs are not primaryconjunction.

Utilitybounds PASSupper≤.010acc-loss/.010NLL/.005summedBrier, VIOLATIONlower>cap, otherwiseUNRESOLVED. PairbenefitPASS lower≥.020, SHORTFALLupper<.020, otherwiseUNRESOLVED. LocalguardPASSupper≤.010 onpairedseed-mean estimand; everyseed reported separately. Components reuse same comparisonintervals with strictpositive lowerbound and eligibility/utility/localrequirements; no extra post-outcomeendpoint or noveltyclaim. Strongest control lowestmeanpairamong predeclared pointeligiblepool,tie≤1e-10frozenlistedorder. No bestseed. Small independent best-ATT-validation pairreader refits JOINT/U_NEW×2seeds(4fits), beforeINNERSEX, secondaryonly.

NLLfloor1e-12; Brier classsum; fixedclassorder, firstargmax; absentclassrecall0 and balancedaccuracy meanall fixedKclasses. Accuracy/NLL/Brier/perclassrecall/confidence and constantbenchmarks beside leakage. Role-optimalconstant is labelled diagnostic, not a deployed test-fitted predictor.

## Resource and closure

One durablequeue/rootsoleexecutor;20CPUh,10elapsed fromqueueepoch,$0cloud,max2heavy acrossmachine including pre-existingPython slot. Onecampaignworker atonce,explicit1numericalthreads,2GiBRSSworker,4GiBaggregate+1GiBOSheadroom. Queuehashpins/processdescendants/CPU/RSS/failures/preemption/outputs. No budget bypass. Publicaggregates/hash/code only; privatepeople/models/readers. Source/runtime safeexport, standalone raw83 labels-not-required, representative backup-alone utility/method/local/SEQ/readers; versioneduncachedlocal andoff-device ifmounted. Drive absencependingcustody, not scientificveto. Atomic paperhandoffs, publiccleanancestry normalpush+remoteverify, no manuscriptmerge/historyrewrite/unrelatedprocessstop.
