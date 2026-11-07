#!/bin/zsh
# usage: seed_chain.sh SEED   (BRIDGE + 4 baselines, then test eval for that seed)
set -e
S=$1; P=.venv/bin/python; L=runs/logs; export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
$P -m bridge.train --name v1_repro_s$S --seed $S > $L/train_v1_repro_s$S.log 2>&1
$P -m experiments.train_baseline --name compgcn_l2_s$S --encoder compgcn --layers 2 --seed $S > $L/train_compgcn_l2_s$S.log 2>&1
$P -m experiments.train_baseline --name compgcn_l6_s$S --encoder compgcn --layers 6 --seed $S > $L/train_compgcn_l6_s$S.log 2>&1
$P -m experiments.train_baseline --name gcn_norel_l6_s$S --encoder gcn --layers 6 --seed $S > $L/train_gcn_norel_l6_s$S.log 2>&1
$P -m experiments.train_baseline --name mlp_text_s$S --encoder mlp --seed $S > $L/train_mlp_text_s$S.log 2>&1
$P -m experiments.eval_baseline --bridge v1_repro_s$S --models compgcn_l2_s$S compgcn_l6_s$S gcn_norel_l6_s$S mlp_text_s$S \
   --out runs/exp/retriever_eval_baselines_s$S.json > $L/eval_baselines_s$S.log 2>&1
echo "seed $S done"
