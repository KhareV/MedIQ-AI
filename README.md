# MedIQ - AI-Powered Medical Consultation Platform

[![Next.js](https://img.shields.io/badge/Next.js-15.2.4-black)](https://nextjs.org/)
[![React](https://img.shields.io/badge/React-19-blue)](https://reactjs.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5-blue)](https://www.typescriptlang.org/)
[![Python](https://img.shields.io/badge/Python-3.8+-green)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-teal)](https://fastapi.tiangolo.com/)

**MedIQ** is an intelligent healthcare assistant that combines advanced Natural Language Processing (NLP), expert medical knowledge systems, and AI-powered analysis to provide comprehensive medical consultation and symptom assessment. This system integrates forward and backward chaining inference engines with modern generative AI to deliver accurate, context-aware medical guidance.

---

## 📋 Table of Contents

- [System Architecture](#-system-architecture)
- [Core Features](#-core-features)
- [Technical Stack](#-technical-stack)
- [AI & NLP System](#-ai--nlp-system)
- [User Interface Components](#-user-interface-components)
- [Medical Records & Context Management](#-medical-records--context-management)
- [Installation & Setup](#-installation--setup)
- [Usage Guide](#-usage-guide)
- [System Documentation](#-system-documentation)

---

## 🏗 System Architecture

MedIQ employs a **hybrid architecture** that combines rule-based expert systems with generative AI:

```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend Layer (Next.js)                  │
│  ┌────────────┐  ┌─────────────┐  ┌──────────────────┐     │
│  │ Chat UI    │  │ Symptom     │  │ Patient          │     │
│  │ Interface  │  │ Analyzer    │  │ Dashboard        │     │
│  └────────────┘  └─────────────┘  └──────────────────┘     │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    API Routes (Next.js)                      │
│  ┌────────────┐  ┌─────────────┐  ┌──────────────────┐     │
│  │ /api/chat  │  │ /api/nlp    │  │ /api/nlp/image   │     │
│  │ (Gemini)   │  │ (Hybrid)    │  │ (Vision)         │     │
│  └────────────┘  └─────────────┘  └──────────────────┘     │
└─────────────────────────────────────────────────────────────┘
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
┌──────────────────────┐    ┌──────────────────────────────┐
│   Gemini AI Engine   │    │  NLP Backend (Python/FastAPI)│
│  - Gemini 2.0 Flash  │    │  ┌────────────────────────┐  │
│  - Medical Context   │    │  │  Hybrid Medical Engine │  │
│  - Chat History      │    │  ├────────────────────────┤  │
└──────────────────────┘    │  │ • Groq AI Integration  │  │
                            │  │ • Forward Chaining     │  │
                            │  │ • Backward Chaining    │  │
                            │  │ • Knowledge Base       │  │
                            │  │ • Consensus Engine     │  │
                            │  └────────────────────────┘  │
                            └──────────────────────────────┘
                                        │
                                        ▼
                            ┌──────────────────────┐
                            │  Expert System KB    │
                            │  • 200+ Conditions   │
                            │  • Medical Rules     │
                            │  • Symptom Mappings  │
                            └──────────────────────┘
```

---

## ✨ Core Features

### 1. **Dual AI Backend System**

- **Gemini 2.0 Flash**: Primary generative AI for natural language understanding
- **NLP Backend**: Python-based expert system with forward/backward chaining inference
- **Seamless Switching**: Real-time toggle between AI backends

### 2. **Intelligent Symptom Analysis**

- Interactive body system selection (Cardiovascular, Respiratory, Neurological, etc.)
- Severity rating and duration tracking
- Vital signs integration (temperature, BP, heart rate)
- Multi-symptom correlation analysis

### 3. **Medical Image Analysis**

- Vision AI-powered medical image interpretation
- Multi-model analysis (LLaMA 11B & 90B)
- Structured findings extraction with confidence levels

### 4. **Expert System Reasoning**

- **Forward Chaining**: Symptom → Diagnosis inference with 200+ medical rules
- **Backward Chaining**: Diagnosis validation and hypothesis testing
- **Syndrome Pattern Recognition**: Clinical pattern matching
- **Inappropriate Diagnosis Filtering**: Context-aware exclusion logic

### 5. **Comprehensive Medical Records**

- Patient medical history integration
- Chronic conditions and allergy tracking
- Current medications database
- Prescription management

### 6. **Interactive 3D Body Visualization**

- Anatomical body model with organ systems
- Interactive symptom localization

### 7. **Personalized Patient Dashboard**

- Health score calculation
- Weekly vital sign trends
- Medication adherence tracking
- Activity monitoring

---

## 🛠 Technical Stack

### Frontend

- **Framework**: Next.js 15.2.4 with App Router
- **UI Library**: React 19 with TypeScript
- **Styling**: Tailwind CSS + shadcn/ui components
- **Animations**: Framer Motion
- **3D Graphics**: Three.js with React Three Fiber
- **State Management**: Zustand with persistence
- **Charts**: Recharts & Nivo

### Backend

- **API Routes**: Next.js API Routes (TypeScript)
- **NLP Server**: FastAPI (Python)
- **AI Models**:
  - Gemini 2.0 Flash (Google)
  - Groq LLaMA 3.3 70B
  - LLaMA Vision models (11B & 90B)

### Database & Storage

- **Database**: Supabase (PostgreSQL)
- **Storage**: Supabase Storage for medical files

---

## 🧠 AI & NLP System

### Hybrid Medical Engine Architecture

The **Hybrid Medical Engine** orchestrates between rule-based reasoning and generative AI:

#### 1. Forward Chaining Inference Engine

Implements symptom-to-diagnosis reasoning with specialized features:

- **200+ Medical Rules** spanning 10+ body systems
- **Diagnostic Patterns**: Biliary syndrome, cardiac syndrome, respiratory syndrome, neurological syndrome
- **Clinical Scoring**: Matches symptoms against expected patterns with confidence weighting
- **Exclusion Logic**: Filters inappropriate diagnoses (e.g., jaundice excludes appendicitis)
- **Urgency Assessment**: Emergency, urgent, moderate, routine classifications

**Example Rule Structure**:

```python
{
  "id": "cholecystitis_001",
  "name": "Cholecystitis",
  "conditions": ["right_upper_quadrant_pain", "fever", "nausea",
                 "pain_after_fatty_foods"],
  "severity": "moderate_to_severe",
  "confidence": 0.85,
  "recommendations": ["emergency_room", "antibiotics", "imaging_studies"]
}
```

#### 2. Backward Chaining Validation Engine

Validates diagnostic hypotheses using evidence-based reasoning:

- **Hypothesis Testing**: Confirms or refutes proposed diagnoses
- **Satisfaction Ratio**: Calculates percentage of expected symptoms present
- **Evidence Modifiers**: Adjusts confidence based on age, duration, severity
- **Multiple Hypothesis Ranking**: Compares and ranks competing diagnoses

#### 3. Medical Knowledge Base

Comprehensive repository of medical knowledge:

- **Coverage**: 200+ conditions across all major body systems
- **Symptom Normalization**: 500+ symptom mappings for natural language
- **Emergency Detection**: Identifies critical conditions requiring immediate care
- **Differential Diagnosis**: Generates ranked list of possible conditions

**Body Systems Covered**:

- Cardiovascular (20+ conditions)
- Respiratory (25+ conditions)
- Gastrointestinal (25+ conditions)
- Neurological (20+ conditions)
- Musculoskeletal (20+ conditions)
- Dermatological (30+ conditions)
- Endocrine (15+ conditions)
- Mental Health (15+ conditions)

#### 4. Consensus Engine

Merges outputs from AI and expert system:

- **Conflict Resolution**: Reconciles disagreements between systems
- **Confidence Harmonization**: Combines confidence scores
- **Priority-based Decision Making**: Weights emergency conditions higher

#### 5. Groq Medical Integration

Processes natural language and medical images:

- **Symptom Extraction**: Identifies symptoms from conversational text
- **Medical Image Analysis**: Interprets X-rays, scans using vision models
- **Context Understanding**: Maintains conversation context across sessions

### Consultation Processing Flow

```
User Input
    ↓
NLP Processing
    ↓
┌───────────────┴───────────────┐
│                               │
Groq AI Analysis        Expert System Analysis
• Extract symptoms      • Forward chaining
• Context understanding • Pattern matching
• Generate insights     • Rule evaluation
│                               │
└───────────────┬───────────────┘
                ↓
        Consensus Engine
        • Merge results
        • Validate findings
        • Prioritize recommendations
                ↓
        Response Generator
        • Structured output
        • Medical disclaimers
        • Follow-up questions
                ↓
        User Response
```

---

## 🎨 User Interface Components

### 1. Chat Interface

Real-time medical consultation with rich features:

- **Dual Backend Support**: Toggle between Gemini and NLP backends
- **Medical History Integration**: Automatically includes patient context
- **File Upload**: Supports medical images and documents
- **Voice Recording**: Audio input capability
- **Structured Formatting**: Organized medical responses with sections
- **Session Persistence**: Maintains chat history using Zustand

**Key Implementation**:

```typescript
// Auto-includes medical context in prompts
const medicalContext = await getMedicalContextForAI();
const response = await generateResponse(medicalContext + userQuery);
```

### 2. Chat Store (State Management)

Zustand-based persistent storage:

```typescript
interface ChatStore {
  chats: Chat[];
  createChat(): string; // Creates new session
  addMessage(chatId, message): void; // Adds message to history
  updateChatTitle(id, title): void; // Auto-names from content
  archiveChat(id): void; // Archive management
  pinChat(id): void; // Priority chats
}
```

**Features**:

- LocalStorage persistence with versioning
- Automatic chat naming from first message
- Archive and pin functionality
- Migration support for schema updates

### 3. Symptom Analyzer

4-step guided symptom assessment:

**Step 1 - Body System Selection**: Visual picker for affected system  
**Step 2 - Symptom Input**: Common symptoms + custom entry  
**Step 3 - Details**: Severity (1-10), duration, vital signs  
**Step 4 - Analysis**: AI-powered differential diagnosis

**Output Format**:

```typescript
{
  condition: "Condition Name",
  probability: 75,                    // % likelihood
  urgency: "low" | "medium" | "high" | "critical",
  description: "Medical description",
  recommendations: string[],
  tests: string[]
}
```

### 4. Medical Records Management

Comprehensive patient data tracking:

```typescript
interface MedicalRecord {
  type: "consultation" | "prescription" | "test" | "diagnosis";
  date: string;
  diagnosis?: string;
  treatment?: string;
  prescription?: string;
  notes?: string;
}
```

**Context Generation**:
Automatically prepares patient context for AI consultations:

- Chronic conditions
- Allergies and sensitivities
- Current medications
- Recent medical history (last 5 records)
- Family history
- Lifestyle factors

### 5. Patient Dashboard

Visual health monitoring interface:

- **Health Score**: Calculated from medical profile and activity
- **Vital Trends**: Charts for heart rate, blood pressure, steps, sleep
- **Medication Adherence**: Bar charts showing compliance rates
- **Activity Breakdown**: Weekly exercise and calorie tracking
- **Quick Actions**: Direct navigation to key features

---

## 📊 Medical Records & Context Management

### Context Integration in AI Consultations

The system automatically enriches AI prompts with patient medical history:

```typescript
const getMedicalContextForAI = async (): Promise<string> => {
  // Aggregates:
  // 1. Medical summary (conditions, allergies, medications)
  // 2. Recent medical history (last 5 records)
  // 3. Active prescriptions
  // 4. Family history

  return `PATIENT MEDICAL CONTEXT:
    MEDICAL SUMMARY:
    - Chronic Conditions: ${conditions.join(", ")}
    - Allergies: ${allergies.join(", ")}
    - Current Medications: ${medications.join(", ")}
    
    RECENT MEDICAL HISTORY:
    ${recentRecords
      .map((r) => `- ${r.date}: ${r.description} (${r.diagnosis})`)
      .join("\n")}
  `;
};
```

This context is prepended to every AI consultation, enabling personalized, history-aware responses.

---

## 🚀 Installation & Setup

### Prerequisites

- Node.js 18+
- Python 3.8+
- npm or yarn

### Quick Start

**1. Clone Repository**

```bash
git clone https://github.com/KhareV/MedIQ-AI.git
cd MedIQ-AI
```

**2. Frontend Setup**

```bash
npm install
cp .env.example .env.local
# Add your API keys to .env.local:
# GEMINI_API_KEY=your_key
# GROQ_API_KEY=your_key
# NLP_SERVER_URL=http://localhost:8000
```

**3. NLP Backend Setup**

```bash
cd nlp
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
# Add API keys to .env
```

**4. Run Development Servers**

Terminal 1 - Frontend:

```bash
npm run dev  # http://localhost:3000
```

Terminal 2 - NLP Backend:

```bash
cd nlp
python app.py  # http://localhost:8000
```

Or run both:

```bash
npm run dev:full
```

---

## 📖 Usage Guide

### Starting a Consultation

1. Navigate to **AI Consultation** from dashboard
2. Select AI backend (Gemini or NLP)
3. Describe symptoms or use suggested prompts
4. Optionally attach medical images
5. Review structured analysis
6. Ask follow-up questions

### Using Symptom Analyzer

1. Select affected body system
2. Add symptoms (common or custom)
3. Rate severity (1-10 scale)
4. Specify duration
5. Input vital signs (optional)
6. Review AI-generated differential diagnosis

### Backend Selection

**Use Gemini AI** for:

- Conversational queries
- General health questions
- Medical education

**Use NLP Backend** for:

- Specific symptom analysis
- Rule-based validation
- Medical image analysis

---

## 📚 System Documentation

### API Endpoints

**Frontend Routes**:

- `POST /api/chat` - Gemini AI consultation
- `POST /api/nlp` - NLP backend medical query
- `POST /api/nlp/image` - Medical image analysis
- `GET /api/nlp` - Health check

**NLP Backend**:

- `POST /api/medical-query` - Process consultation
- `POST /api/validate-diagnosis` - Validate hypothesis
- `POST /api/analyze-image` - Analyze medical images
- `GET /api/health` - Health check
- `WebSocket /ws/medical-chat/{id}` - Real-time chat

### Medical Knowledge Base Statistics

- **Total Conditions**: 200+
- **Emergency Conditions**: 10+
- **Symptom Mappings**: 500+
- **Body Systems**: 10
- **Diagnostic Patterns**: 4 specialized syndromes

---

## 🔒 Medical Disclaimers

⚠️ **IMPORTANT NOTICES**

- **Educational Purpose Only**: This system is for informational use
- **Not Medical Advice**: Always consult qualified healthcare professionals
- **Emergency Situations**: Call 911 for urgent medical needs
- **AI Limitations**: AI can make errors; verify information
- **Privacy**: Review privacy policy before entering personal data

---

## 👥 Authors

**Project**: MedIQ AI Medical Consultation Platform  
**Developer**: KhareV  
**Repository**: [github.com/KhareV/MedIQ-AI](https://github.com/KhareV/MedIQ-AI)

---

## 🙏 Acknowledgments

- Google Gemini for generative AI
- Groq for LLM inference
- Medical knowledge databases
- Open source community (React, Next.js, FastAPI)

---

**Built for better healthcare accessibility through AI**

_Version 2.0 • November 2025_
