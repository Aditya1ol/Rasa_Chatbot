document.addEventListener("DOMContentLoaded", () => {
  const chatMessages = document.getElementById("chat-messages");
  const chatForm = document.getElementById("chat-form");
  const userInput = document.getElementById("msg");
  const questionDropdown = document.getElementById("question-dropdown");
  const sendDropdownBtn = document.getElementById("send-dropdown-btn");

  // Function to append message to chat
  function appendMessage(message, sender = "bot") {
    const messageDiv = document.createElement("div");
    messageDiv.classList.add("message", sender);
    messageDiv.textContent = message;
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }

  // Send message to backend and display response
  async function sendMessage(message) {
    appendMessage(message, "user");
    try {
      const response = await fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message }),
      });
      const data = await response.json();
      if (data.response && data.response.length > 0) {
        data.response.forEach((resp) => appendMessage(resp, "bot"));
      } else {
        appendMessage("Sorry, I couldn't find any relevant information.", "bot");
      }
    } catch (error) {
      appendMessage("Error communicating with server.", "bot");
    }
  }

  // Handle form submit (text input)
  chatForm.addEventListener("submit", (e) => {
    e.preventDefault();
    const message = userInput.value.trim();
    if (message) {
      sendMessage(message);
      userInput.value = "";
    }
  });

  // Handle dropdown ask button click
  sendDropdownBtn.addEventListener("click", () => {
    const selectedQuestion = questionDropdown.value;
    if (selectedQuestion) {
      sendMessage(selectedQuestion);
    }
  });
});
