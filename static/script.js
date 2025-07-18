// 📦 Wait for page content to load before attaching logic
document.addEventListener("DOMContentLoaded", () => {
  const chatMessages = document.getElementById("chat-messages");
  const chatForm = document.getElementById("chat-form");
  const userInput = document.getElementById("msg");
  const questionDropdown = document.getElementById("question-dropdown");
  const sendDropdownBtn = document.getElementById("send-dropdown-btn");

  let lastSentMessage = "";

  /**
   * 🧱 Append a message to the chat window
   * @param {string} message - The message to display
   * @param {string} sender - 'user' or 'bot'
   */
  function appendMessage(message, sender = "bot") {
    const messageDiv = document.createElement("div");
    messageDiv.classList.add("message", sender);
    messageDiv.textContent = message;
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }

  /**
   * 🚀 Send user message to backend Flask API
   * @param {string} message - User's input or dropdown selection
   */
 async function sendMessage(message) {
  if (!message) return;
  appendMessage(message, "user");

  try {
    const response = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message })
    });

    const data = await response.json();
    if (data.response && Array.isArray(data.response)) {
      data.response.forEach(resp => {
        if (resp) appendMessage(resp, "bot");
      });
    } else {
      appendMessage("🤖 No valid response from server.", "bot");
    }
  } catch (error) {
    console.error("Error:", error);
    appendMessage("⚠️ Server error occurred.", "bot");
  }

  userInput.value = "";


    // Allow asking same message again after 3 seconds
    setTimeout(() => {
      lastSentMessage = "";
    }, 3000);
  }

  /**
   * ✉️ Handle form submit (text input send)
   */
  chatForm.addEventListener("submit", (e) => {
    e.preventDefault();
    const message = userInput.value.trim();
    if (message) {
      sendMessage(message);
    }
  });

  /**
   * 📥 Handle dropdown 'Ask' button click
   */
  

  /**
   * 📝 Feedback submission logic
   */
  window.submitFeedback = async function () {
    const feedback = document.getElementById("feedbackInput").value.trim();
    const status = document.getElementById("feedbackStatus");

    if (!feedback) return;

    try {
      const res = await fetch("/submit_feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ feedback })
      });

      const data = await res.json();
      status.innerText = data.message;
      document.getElementById("feedbackInput").value = "";
    } catch (err) {
      status.innerText = "⚠️ Could not send feedback.";
      console.error("Feedback error:", err);
    }
  };
});
