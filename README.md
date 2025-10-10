# Typing-Biometrics-Authentication

The vulnerability of traditional password-
based systems underscores a critical need for robust al-
ternative authentication mechanisms. Keystroke dynamics,

which captures the unique rhythmic patterns of an individ-
ual’s typing, presents a powerful behavioral biometric for

this purpose. Although numerous authentication methods
leveraging this data exist, a predominant limitation is their
inability to generalize effectively from a minimal number of
user samples. To address this one-shot learning challenge,
we introduce a novel two-stage framework. Initially, a deep
metric learning model, inspired by FaceNet [Schroff et al.,

2015], projects raw keystroke sequences into a discrimi-
native low-dimensional embedding space. Subsequently, a

decoder neural network is trained to verify user identity
by analyzing the relationships between these embeddings.
This architecture is specifically designed to enable reliable
authentication for new users with limited enrolled data.
