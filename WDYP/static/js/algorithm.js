const models = [
  "logistic_regression",
  "random_forest",
  "svm_rbf",
  "decision_tree",
  "gradient_boosting",
  "knn_euclidean",
  "naive_bayes",
  "linear_discriminant_analysis",
  "adaboost",
  "mlp_neural_network",
];
const container = document.getElementById("image-container");

// Loop through each model and create an image element
models.forEach((model) => {
  const img = document.createElement("img");
  img.src = `/static/confusion_matrix/${model}_confusion_matrix.png`;
  console.log(img.src);
  img.alt = model;
  img.style.width = "50%";
  container.appendChild(img);
});
window.addEventListener("DOMContentLoaded", (event) => {
  // Automatically close the alert after 5 seconds (5000 milliseconds)
  setTimeout(closeAlerts, 5000);
});

function closeAlerts() {
  let alerts = document.querySelectorAll(".alert");
  alerts.forEach((alert) => {
    alert.style.display = "none";
  });
}

// Function to add a new row to the table
function addNewRow() {
  const row = `
  <tr>
  <td><input type="number" step="0.01" min="0" max="100" class="form-control" name="n" value="0"></td>
  <td><input type="number" step="0.01" min="0" max="100" class="form-control" name="p" value="0"></td>
  <td><input type="number" step="0.01" min="0" max="100" class="form-control" name="k" value="0"></td>
  <td><input type="number" step="0.01" min="-40" max="50" class="form-control" name="temperature" value="0"></td>
  <td><input type="number" step="0.01" min="0" max="100" class="form-control" name="humidity" value="0"></td>
  <td><input type="number" step="0.01" min="1" max="14" class="form-control" name="ph" value="7"></td>
  <td><input type="number" step="0.01" min="0" max="1000" class="form-control" name="rainfall" value="0"></td>
  <td><button class="btn btn-danger remove-row">Remover</button></td>
</tr>
        `;
  document.getElementById("inputRows").insertAdjacentHTML("beforeend", row);
  updateTableDisplayStyle();
}

document.addEventListener("DOMContentLoaded", function() {
  const form = document.querySelector("form");
  
  form.addEventListener("submit", function(event) {
      // Verifica todos os inputs do tipo 'number'
      const numberInputs = form.querySelectorAll("input[type='number']");
      
      for (let input of numberInputs) {
          const min = parseFloat(input.getAttribute("min"));
          const max = parseFloat(input.getAttribute("max"));
          const value = parseFloat(input.value);
          
          if (value < min || value > max) {
              alert(`Por favor, insira um valor válido para ${input.name}.`);
              event.preventDefault(); // Impede o envio do formulário
              return;
          }
      }
  });
});

// Function to remove a row from the table
function removeRow(event) {
  if (event.target.classList.contains("remove-row")) {
    event.target.closest("tr").remove();
    updateTableDisplayStyle();
  }
}

// Function to update input fields based on selected instance
function updateInputFields() {
  const selectElement = document.getElementById("instanceSelect");
  const selectedInstanceId = selectElement.value;

  if (selectedInstanceId) {
    const instanceData = {
      name: "{{ selected_instance.name }}",
      author: "{{ selected_instance.author }}",
    };

    document.getElementById("instanceName").value = instanceData.name;
    document.getElementById("instanceAuthor").value = instanceData.author;
  } else {
    document.getElementById("instanceName").value = "";
    document.getElementById("instanceAuthor").value = "";
  }
}

// Function to update table display style
function updateTableDisplayStyle() {
  const table = document.querySelector(".table");
  const rowCount = table.querySelectorAll("tbody tr").length;

  if (rowCount > 0) {
    table.classList.add("table-scrollable");
  } else {
    table.classList.remove("table-scrollable");
  }
}

// Initial call to update table display
updateTableDisplayStyle();

// Event listeners
document.getElementById("addRow").addEventListener("click", addNewRow);
document.addEventListener("click", removeRow);
