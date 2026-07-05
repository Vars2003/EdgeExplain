# EdgeExplain Phase 6.2 Implementation Tasks

- [ ] Federated Learning Package Enhancements
  - [ ] Modify `federated/config.py` (Add DEFAULT_EPOCHS and DEFAULT_AGGREGATOR values)
  - [ ] Modify `federated/aggregator.py` (Implement true sample-weighted averaging for Logistic Regression and unsupported stubs for tree models)
  - [ ] Modify `federated/models.py` (Add weight extractors/setters for Scikit-Learn models)
  - [ ] Modify `federated/trainer.py` (Implement single fit local training steps per round)
  - [ ] Modify `federated/client.py` (Receive global weights, run local training, export parameters)
  - [ ] Modify `federated/server.py` (Coordinate multi-round broadcast-train-aggregate loop)
  - [ ] Modify `federated/communication.py` (Compute estimated bytes transferred and track network delay)
  - [ ] Modify `federated/metrics.py` (Log communication_history, convergence_history, and round performance indicators)
- [ ] Memory Object Integration
  - [ ] Modify `intelligence/memory.py` to append the refined `aggregation` section, global accuracy, global loss, and convergence logs
- [ ] Streamlit Dashboard View
  - [ ] Modify `pages/15_Federated_Learning.py` (Interactive training panel, 5 Plotly charts, client lists, and tree model warning banners)
- [ ] Testing & Verification
  - [ ] Update `scratch/test_runner.py` with Phase 6.2 test cases (weighted average accuracy checks, metric serializations)
  - [ ] Execute tests to verify training convergence and correct unsupported stubs
  - [ ] Update `walkthrough.md` with execution summaries
