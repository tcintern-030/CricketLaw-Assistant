const questionInput =
    document.getElementById("question");

const askButton =
    document.getElementById("askButton");

const loading =
    document.getElementById("loading");

const answerSection =
    document.getElementById("answerSection");

const answer =
    document.getElementById("answer");

const sourcesSection =
    document.getElementById("sourcesSection");

const sources =
    document.getElementById("sources");

const suggestionButtons =
    document.querySelectorAll(".suggestion");


/* ASK BUTTON */

askButton.addEventListener(
    "click",
    askQuestion
);


/* SUGGESTIONS */

suggestionButtons.forEach(button => {

    button.addEventListener("click", () => {

        questionInput.value =
            button.textContent.trim();

        questionInput.focus();

    });

});


/* ENTER KEY */

questionInput.addEventListener(
    "keydown",
    event => {

        if (
            event.key === "Enter" &&
            !event.shiftKey
        ) {

            event.preventDefault();

            askQuestion();

        }

    }
);


/* ASK QUESTION */

async function askQuestion() {

    const question =
        questionInput.value.trim();


    if (!question) {

        questionInput.focus();

        return;

    }


    /* UI */

    askButton.disabled = true;

    loading.classList.remove("hidden");

    answerSection.classList.add("hidden");

    sourcesSection.classList.add("hidden");


    try {

        const response =
            await fetch("/ask", {

                method: "POST",

                headers: {
                    "Content-Type":
                        "application/json"
                },

                body: JSON.stringify({
                    question: question
                })

            });


        const data =
            await response.json();


        if (!response.ok) {

            throw new Error(
                data.detail ||
                "Unable to process your question."
            );

        }


        /* ANSWER */

        answer.textContent =
            data.answer || "No answer available.";

        answerSection.classList.remove(
            "hidden"
        );


        /* SOURCES */

        sources.innerHTML = "";


        if (
            data.sources &&
            data.sources.length > 0
        ) {

            data.sources.forEach(
                (source, index) => {

                    const card =
                        document.createElement(
                            "div"
                        );

                    card.className =
                        "source-card";


                    card.innerHTML = `

                        <div class="source-top">

                            <span class="source-number">
                                SOURCE ${index + 1}
                            </span>

                            <span class="source-page">
                                ${
                                    source.page
                                        ? "Page " +
                                          source.page
                                        : ""
                                }
                            </span>

                        </div>

                        <div class="source-content">
                            ${
                                source.content ||
                                "Retrieved MCC section."
                            }
                        </div>

                    `;


                    sources.appendChild(card);

                }
            );


            sourcesSection.classList.remove(
                "hidden"
            );

        }


        /* Scroll to answer */

        answerSection.scrollIntoView({
            behavior: "smooth",
            block: "start"
        });


    } catch (error) {

        answer.textContent =
            error.message;

        answerSection.classList.remove(
            "hidden"
        );

    } finally {

        loading.classList.add("hidden");

        askButton.disabled = false;

    }

}