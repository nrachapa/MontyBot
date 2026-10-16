# Terminal-only UCI play

The model is played through a terminal UCI bridge over the existing ONNX/ORT inference path. We rejected building a custom GUI (and custom board renderers) as feature work that spends effort away from model strength; UCI is the one adapter that also plugs into any external UCI GUI for free if the owner ever wants one.
