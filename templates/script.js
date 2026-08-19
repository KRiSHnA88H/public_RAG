const questionInput =
    document.getElementById("question");

const askButton =
    document.getElementById("askBtn");

const chat =
    document.getElementById("chat");


async function askQuestion() {

    const question =
        questionInput.value.trim();


    if (question === "") {
        return;
    }


    chat.innerHTML += `
        <div>
            <strong>You:</strong>
            ${question}
        </div>
    `;


    questionInput.value = "";


    chat.innerHTML += `
        <div id="loading">
            <strong>AI:</strong>
            Thinking...
        </div>
    `;


    try {

        const response = await fetch("/ask", {

            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify({
                question: question
            })

        });


        const data =
            await response.json();


        document
            .getElementById("loading")
            .remove();


        chat.innerHTML += `
            <div>
                <strong>AI:</strong>
                ${data.answer}
            </div>
        `;

    }


    catch (error) {

        document
            .getElementById("loading")
            .remove();


        chat.innerHTML += `
            <div>
                <strong>Error:</strong>
                ${error.message}
            </div>
        `;

    }

}


askButton.addEventListener(
    "click",
    askQuestion
);


questionInput.addEventListener(
    "keydown",
    function(event) {

        if (event.key === "Enter") {
            askQuestion();
        }

    }
);