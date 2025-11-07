import { type NextRequest, NextResponse } from "next/server";
import Groq from "groq-sdk";

const groq = new Groq({
  apiKey: process.env.GROQ_API_KEY || "",
});

export async function POST(request: NextRequest) {
  try {
    const { message, medicalContext } = await request.json();

    if (!message) {
      return NextResponse.json(
        { error: "Message is required" },
        { status: 400 }
      );
    }

    if (!process.env.GROQ_API_KEY) {
      return NextResponse.json(
        {
          response:
            "I'm sorry, but I need a Groq API key to function properly. Please add your GROQ_API_KEY to the environment variables in your project settings.",
        },
        { status: 200 }
      );
    }

    const enhancedPrompt = `You are MedIQ AI, a sophisticated medical consultation assistant. You provide evidence-based healthcare guidance while emphasizing the importance of professional medical care.

IMPORTANT GUIDELINES:
- Always recommend consulting healthcare professionals for diagnosis and treatment
- Provide educational information, not definitive medical advice
- Be empathetic and supportive in your responses
- Use clear, accessible language while maintaining medical accuracy
- Acknowledge limitations and encourage professional consultation

${medicalContext ? `PATIENT MEDICAL CONTEXT:\n${medicalContext}\n\n` : ""}

PATIENT QUERY: ${message}

Please provide a helpful, informative response that considers the patient's medical history (if provided) while maintaining appropriate medical disclaimers.`;

    const chatCompletion = await groq.chat.completions.create({
      messages: [
        {
          role: "user",
          content: enhancedPrompt,
        },
      ],
      model: "llama-3.3-70b-versatile",
      temperature: 0.7,
      max_tokens: 2048,
    });

    const text =
      chatCompletion.choices[0]?.message?.content ||
      "I apologize, but I couldn't generate a response.";

    return NextResponse.json({ response: text });
  } catch (error: any) {
    console.error("Error generating response:", error);

    let errorMessage =
      "I apologize, but I encountered an error while processing your request. Please try again.";

    if (
      error?.message?.includes("API_KEY_INVALID") ||
      error?.message?.includes("Invalid API Key")
    ) {
      errorMessage =
        "The Groq API key appears to be invalid. Please check your GROQ_API_KEY environment variable.";
    } else if (
      error?.message?.includes("QUOTA_EXCEEDED") ||
      error?.message?.includes("quota")
    ) {
      errorMessage =
        "I've reached my usage limit for now. Please try again later or check your Groq API quota.";
    } else if (
      error?.message?.includes("RATE_LIMIT_EXCEEDED") ||
      error?.message?.includes("rate limit")
    ) {
      errorMessage =
        "I'm receiving too many requests right now. Please wait a moment and try again.";
    } else if (error?.message?.includes("SAFETY")) {
      errorMessage =
        "I can't provide a response to that request due to safety guidelines. Please try rephrasing your question.";
    }

    return NextResponse.json({ response: errorMessage }, { status: 200 });
  }
}
