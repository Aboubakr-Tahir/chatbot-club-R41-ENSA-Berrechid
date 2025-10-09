import { useState, useEffect, useRef } from "react";
import { marked } from "marked";

function App() {
  // 'useState' is a React Hook to manage data that changes over time.
  // 'messages' will hold our array of chat messages.
  const [messages, setMessages] = useState([
    {
      role: "bot",
      content: "Hello! I am the R41 chatbot. How can I help you?",
    },
  ]);
  // 'input' will hold the text the user is currently typing.
  const [input, setInput] = useState("");
  const messagesEndRef = useRef(null);

  // This function automatically scrolls the chat to the latest message.
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  // 'useEffect' is a Hook that runs code in response to data changes.
  // This one runs 'scrollToBottom' whenever the 'messages' array is updated.
  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // This function is called when the user submits the form.
  const handleSend = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    const userMessage = { role: "user", content: input };
    // This is the most up-to-date history BEFORE the bot responds.
    const updatedMessagesWithUser = [...messages, userMessage];

    setMessages(updatedMessagesWithUser); // Add user's message to the UI
    setInput(""); // Clear the input box

    // Immediately add a placeholder for the bot's response for a better UX
    setMessages((prev) => [...prev, { role: "bot", content: "..." }]);

    try {
      const response = await fetch("http://localhost:8000/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: input,
          // *** THE FIX IS HERE ***
          // Send the complete history UP TO the user's current question.
          chat_history: updatedMessagesWithUser.map((msg) => ({
            role: msg.role === "user" ? "human" : "ai",
            content: msg.content,
          })),
        }),
      });

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let fullResponse = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        fullResponse += decoder.decode(value);

        // Update the last message (the bot's placeholder) with the streaming content
        setMessages((currentMessages) => {
          const updatedMessages = [...currentMessages];
          updatedMessages[updatedMessages.length - 1].content = fullResponse;
          return updatedMessages;
        });
      }
    } catch (error) {
      console.error("Failed to fetch chat response:", error);
      setMessages((currentMessages) => {
        const updatedMessages = [...currentMessages];
        updatedMessages[updatedMessages.length - 1].content =
          "Sorry, I couldn't connect to the server.";
        return updatedMessages;
      });
    }
  };

  // This is the JSX that defines what our component looks like.
  // It uses Tailwind CSS classes for all styling.
  return (
    <div className="flex flex-col h-screen bg-gray-100">
      <div className="flex-grow p-4 overflow-auto">
        <div className="max-w-3xl mx-auto">
          {messages.map((msg, index) => (
            <div
              key={index}
              className={`flex ${
                msg.role === "user" ? "justify-end" : "justify-start"
              } mb-4`}
            >
              <div
                className={`max-w-lg p-3 rounded-lg shadow-md ${
                  msg.role === "user"
                    ? "bg-blue-500 text-white"
                    : "bg-white text-black"
                }`}
                // Use dangerouslySetInnerHTML to render the HTML from 'marked'
                dangerouslySetInnerHTML={{ __html: marked.parse(msg.content) }}
              />
            </div>
          ))}
          <div ref={messagesEndRef} />
        </div>
      </div>
      <div className="p-4 bg-white border-t">
        <form onSubmit={handleSend} className="max-w-3xl mx-auto flex">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask about R41..."
            className="flex-grow px-4 py-2 border rounded-l-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
          <button
            type="submit"
            className="px-4 py-2 bg-blue-500 text-white rounded-r-lg hover:bg-blue-600 focus:outline-none"
          >
            Send
          </button>
        </form>
      </div>
    </div>
  );
}

export default App;
