# Architecture Mindmap

```mermaid
mindmap
  root((Diffusion-Guided Docking))
    Core Idea
      DiffGui Inspiration
        Differentiable Auxiliary Network
        Gradient-Guided Updates
        Real-Time Trajectory Correction
      Extension to Docking
        Bond Prediction → Docking Score
        Geometric Constraints → Docking Constraints
    Model Architecture
      Base Diffusion Model
        FlashDiff
          3D Coordinate Generation
          Atom Type Generation
          Bond Type Generation
      Geometry Guidance Module
        Pocket Center Guidance
          Energy Function Design
          Gradient Computation
          Coordinate Update
        Collision Penalty
          Minimum Distance Constraint
          Energy Penalty Term
      Docking Score Guidance
        DockingScorePredictor
          Network Architecture
            Atom Type Embeddings
            Distance Feature Extraction
            Pair-Level Feature Fusion
          Training Data
            MOAD Dataset
            Vina Docking Scores
          Gradient Guidance
            Score Computation
            Gradient Ascent
            Coordinate Update
    Two-Stage Strategy
      Sampling Phase
        Stage A: Geometry Guidance
          Time Step t > T/2
          Fast Pocket Localization
          No Training Data Required
        Stage B: Docking Guidance
          Time Step t <= T/2
          Fine-Tune Binding Modes
          Requires Trained Predictor
      Design Advantages
        Geometry Guidance Fast Convergence
        Docking Guidance Precise Optimization
        Balance Efficiency and Quality
    Technical Details
      Differentiability
        Coordinates requires_grad
        Gradient Backpropagation
        Automatic Differentiation
      Guidance Strength Control
        alpha_geo
        alpha_dock
        Step Size Adjustment
      Energy Function Design
        Center Guidance Term
        Collision Penalty Term
        Weight Balancing
    Experimental Validation
      Dataset
        MOAD
        Protein-Ligand Pairs
        Docking Labels
      Baseline Comparison
        No Guidance Baseline
        Geometry Only
        Docking Only
        Two-Stage Guidance
      Performance Evaluation
        Docking Score Comparison
        Success Rate Statistics
        Computational Efficiency
    Key Innovations
      Concept Transfer
        DiffGui's Diffusion Guidance
        Bond Prediction → Docking Score
        Real-Time Guidance → Post-Hoc Screening
      Technical Implementation
        Differentiable Docking Network
        Approximate Vina Scores
        End-to-End Training
      Two-Stage Guidance
        Geometry Fast Localization
        Docking Precise Optimization
    Design Principles
      Differentiability
        All Components Differentiable
        Gradients Propagatable
      Real-Time
        Guidance During Diffusion
        Not Post-Hoc Screening
      Flexibility
        Multiple Guidance Modes
        Combinable Design
```

## Architecture Flow

```mermaid
graph TB
    A[Input: Protein Pocket] --> B[Base Diffusion Model]
    B --> C[Diffusion Sampling Loop t=T→0]
    
    C --> D{Time Step t}
    D --> E[Standard Diffusion Update]
    E --> F[Predict x_{t-1}]
    
    F --> G{Guidance Strategy}
    G -->|Stage A: Geometry| H[Geometry Energy]
    G -->|Stage B: Docking| I[Docking Score]
    
    H --> H1[Pocket Center Energy]
    H --> H2[Collision Penalty]
    H1 --> H3[Gradient Computation]
    H2 --> H3
    H3 --> H4[Coordinate Update]
    
    I --> I1[DockingScorePredictor]
    I1 --> I2[Protein-Ligand Features]
    I2 --> I3[Distance Encoding RBF]
    I3 --> I4[Pair Feature Fusion]
    I4 --> I5[Pooling and Prediction]
    I5 --> I6[Docking Score]
    I6 --> I7[Gradient Ascent Update]
    
    H4 --> J[Updated x_{t-1}]
    I7 --> J
    
    J --> K{t > 0?}
    K -->|Yes| C
    K -->|No| L[Final Ligand Structure]
    
    L --> M[Docking Evaluation]
    M --> N[Output Results]
```

## DiffGui Comparison

```mermaid
graph LR
    subgraph DiffGui Original
        A1[Bond Predictor] --> A2[Predict Bond Probabilities]
        A2 --> A3[Construct Objective<br/>Entropy/Logit/CrossEnt]
        A3 --> A4[Compute Gradient]
        A4 --> A5[Update Coordinates<br/>delta = -alpha * grad]
    end
    
    subgraph Our Docking Design
        B1[DockingScorePredictor] --> B2[Predict Docking Score]
        B2 --> B3[Construct Objective<br/>Score Maximization]
        B3 --> B4[Compute Gradient]
        B4 --> B5[Update Coordinates<br/>delta = +alpha * grad]
    end
    
    A1 -.->|Concept Transfer| B1
    A3 -.->|Same Design Pattern| B3
    A5 -.->|Same Update Mechanism| B5
```

