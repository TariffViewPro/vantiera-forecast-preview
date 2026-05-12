let forecastChart = null;
const MAX_PREVIEW_RUNS = 3;

const forecastButton = document.getElementById("forecastButton");

forecastButton.addEventListener("click", async () => {

    const runsUsed = parseInt(
        localStorage.getItem("forecast_preview_runs") || "0"
    );

    if (runsUsed >= MAX_PREVIEW_RUNS) {

        const errorBox = document.getElementById("errorBox");

        errorBox.innerHTML = `
            <strong>Forecast preview limit reached.</strong><br>
            For ongoing forecasting workflows, explore the full Vantiera platform.
        `;

        errorBox.classList.remove("hidden");

        return;
    }    

    const historyInput = document.getElementById("historyInput").value;

    const errorBox = document.getElementById("errorBox");
    const resultsSection = document.getElementById("resultsSection");
    const summaryNote = document.getElementById("summaryNote");
    const tableBody = document.querySelector("#forecastTable tbody");
    const tableHead = document.querySelector("#forecastTable thead");

    errorBox.classList.add("hidden");
    resultsSection.classList.add("hidden");

    forecastButton.disabled = true;
    forecastButton.textContent = "Generating...";

    tableHead.innerHTML = "";
    tableBody.innerHTML = "";

    try {

        const response = await fetch("/forecast-preview", {

            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify({
                history: historyInput
            })

        });

        if (!response.ok) {

            if (response.status === 429) {
                throw new Error(
                    "Unable to process additional requests."
                );
            }

            throw new Error("Server error.");
        }

        const data = await response.json();

        if (data.status !== "success") {
            throw new Error(data.message);
        }

        const result = data.result;

        summaryNote.textContent = result.summary_note || "";

        const historyLabels = [];

        for (let i = 0; i < result.history.length; i++) {
            historyLabels.push(`H${i + 1}`);
        }

        const forecastLabels = [];

        for (let i = 0; i < result.forecast.length; i++) {
            forecastLabels.push(`F${i + 1}`);
        }

        const allLabels = [...historyLabels, ...forecastLabels];

        const historyData = [
            ...result.history,
            ...new Array(result.forecast.length).fill(null)
        ];

        const forecastData = [
            ...new Array(result.history.length - 1).fill(null),
            result.history[result.history.length - 1],
            ...result.forecast
        ];

        const ctx = document
            .getElementById("forecastChart")
            .getContext("2d");

        if (forecastChart) {
            forecastChart.destroy();
        }

        forecastChart = new Chart(ctx, {

            type: "line",

            data: {

                labels: allLabels,

                datasets: [

                    {
                        label: "History",
                        data: historyData,
                        borderColor: "#808080",
                        backgroundColor: "#808080",
                        borderWidth: 2,
                        pointRadius: 3,
                        pointHoverRadius: 4,
                        tension: 0.2
                    },

                    {
                        label: "Forecast",
                        data: forecastData,
                        borderColor: "#1F5FAF",
                        backgroundColor: "#1F5FAF",
                        borderWidth: 2,
                        pointRadius: 3,
                        pointHoverRadius: 4,
                        tension: 0.2
                    }

                ]
            },

            options: {

                responsive: true,
                maintainAspectRatio: false,

                plugins: {

                    legend: {
                        position: "bottom",
                        labels: {
                            usePointStyle: true,
                            boxWidth: 8
                        }
                    }
                },

                scales: {

                    y: {
                        grid: {
                            color: "#E5E7EB"
                        },
                        ticks: {
                            color: "#6B7280"
                        }
                    },

                    x: {
                        grid: {
                            display: false
                        },
                        ticks: {
                            color: "#6B7280"
                        }
                    }
                }
            }
        });

        const headerRow = document.createElement("tr");
        const valueRow = document.createElement("tr");

        headerRow.innerHTML = `<th>Forecast</th>`;
        valueRow.innerHTML = `<td>Units</td>`;

        for (let i = 0; i < result.forecast.length; i++) {

            headerRow.innerHTML += `<th>M${i + 1}</th>`;

            valueRow.innerHTML += `
                <td>${result.forecast[i]}</td>
            `;
        }

        tableHead.appendChild(headerRow);
        tableBody.appendChild(valueRow);

        resultsSection.classList.remove("hidden");

        localStorage.setItem(
            "forecast_preview_runs",
            runsUsed + 1
        );

    } catch (error) {

        errorBox.textContent = error.message;
        errorBox.classList.remove("hidden");

    } finally {

        forecastButton.disabled = false;
        forecastButton.textContent = "Generate Forecast";
    }

});