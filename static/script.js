$(document).ready(function () {
  console.log("✅ script.js 已成功載入！");

   /* 1. 頁面載入後先抓取專案列表，填入下拉選單*/
  loadProjectList();
  function loadProjectList() {
    $.ajax({
      type: "GET",
      url: "/list_projects",
      success: function (projects) {
        let $select = $("#project-list");
        $select.empty();
        $select.append(`<option value="">選擇專案</option>`);
        projects.forEach(proj => {
          $select.append(`<option value="${proj}">${proj}</option>`);
        });
      },
      error: function (err) {
        console.warn("❌ 無法取得專案列表：", err);
      }
    });
  }

   /* 2. 點擊「載入專案」 → 讀取 saved_projects/xxx 內的 points.csv 和 bounds.csv*/
  $("#load-project-btn").click(function(){
    let projName = $("#project-list").val();
    if(!projName){
      alert("請先選擇專案！");
      return;
    }
    $.ajax({
      type: "GET",
      url: "/load_project?project_name=" + encodeURIComponent(projName),
      success: function(res) {
        // 假設後端回傳 { pointsCsv: "...", boundsCsv: "..." }
        parseAndFillPointsCsv(res.pointsCsv);
        parseAndFillBoundsCsv(res.boundsCsv);
        // 將專案名稱自動填入 #project-name
        $("#project-name").val(projName);
      },

      error: function(err){
        alert("❌ 載入專案失敗：" + (err.responseJSON?.error || "未知錯誤"));
      }
    });
  });

  // 解析 points.csv 字串，並填入表格
  function parseAndFillPointsCsv(pointsCsv) {
    let lines = pointsCsv.trim().split("\n");
    if(lines.length < 2) {
      console.warn("points.csv 沒有足夠行數");
      return;
    }
    let headerRow = lines[0].split(",");
    let totalCols = headerRow.length; // xCount + 1(y)
    if(totalCols < 2){
      console.warn("points.csv: 至少 1個X + 1個Y");
      return;
    }
    let xHeaders = headerRow.slice(0, totalCols-1);
    let yHeader = headerRow[totalCols-1];

    // 先更新表頭 & bounds-row
    updateTableHeaders(xHeaders, yHeader);

    // 其餘行 => data-row
    $(".data-row").remove();
    let dataLines = lines.slice(1);
    dataLines.forEach(line => {
      let cols = line.split(",");
      if(cols.length < totalCols) return; // 格式不符

      let tr = $("<tr>").addClass("data-row");
      // X
      for(let i=0; i < totalCols-1; i++){
        tr.append($("<td>").append(
          $("<input>").attr("type","number").attr("step","0.1").val(cols[i].trim())
        ));
      }
      // Y
      tr.append($("<td>").append(
        $("<input>").attr("type","number").attr("step","0.1").val(cols[totalCols-1].trim())
      ));
      // 刪除
      tr.append($("<td>").append($("<button>").addClass("delete-row").text("X")));
      $("#bounds-row").before(tr);
    });
  }

  // 解析 bounds.csv 字串，並填入 bounds-row
  function parseAndFillBoundsCsv(boundsCsv) {
    // bounds.csv:
    //   第一行 => x1,x2,x3,...
    //   第二行 => 下界
    //   第三行 => 上界
    let lines = boundsCsv.trim().split("\n");
    if(lines.length < 3){
      console.warn("bounds.csv 行數不足：", lines.length);
      return;
    }
    let xHeaders = lines[0].split(",");
    let lowers = lines[1].split(",");
    let uppers = lines[2].split(",");

    if(xHeaders.length !== lowers.length || xHeaders.length !== uppers.length){
      console.warn("bounds.csv 欄位數對不上");
      return;
    }

    // 先清空 bounds-row
    let boundsRow = $("#bounds-row");
    boundsRow.empty();

    // 依 xHeaders 產生對應的 <td>
    xHeaders.forEach((h, idx) => {
      let lowerVal = lowers[idx].trim();
      let upperVal = uppers[idx].trim();

      let td = $("<td>").append(
        $("<div>").addClass("bounds-container").append(
          $("<label>").addClass("bound-label").text("上界"),
          $("<input>").attr("type","number").attr("step","0.1").addClass("bound-max").val(upperVal)
        )
      ).append(
        $("<div>").addClass("bounds-container").append(
          $("<label>").addClass("bound-label").text("下界"),
          $("<input>").attr("type","number").attr("step","0.1").addClass("bound-min").val(lowerVal)
        )
      );
      boundsRow.append(td);
    });
    // 最後補 2 <td> 給 y & 刪除
    boundsRow.append("<td></td><td></td>");
  }

 /* updateTableHeaders(xHeaders, yHeader) + 同步建 bounds-row*/
function updateTableHeaders(xHeaders, yHeader) {
    // 保存當前的上下界值
    const savedBounds = {};
    $("#bounds-row td").each(function(index) {
        const headerText = $("#header-row th").eq(index).text().trim();
        const min = $(this).find(".bound-min").val();
        const max = $(this).find(".bound-max").val();
        if (min !== "" || max !== "") {
            savedBounds[headerText] = {
                min: min !== "" ? parseFloat(min).toFixed(2) : "", // 格式化為 2 位小數
                max: max !== "" ? parseFloat(max).toFixed(2) : ""  // 格式化為 2 位小數
            };
        }
    });

    // 更新表頭
    let headerRow = $("#header-row");
    headerRow.empty();

    // X 欄
    xHeaders.forEach(h => {
        headerRow.append(
            $("<th>").attr("data-x-col", "true")
                     .attr("contenteditable", "true")
                     .text(h)
        );
    });
    
    // Y 欄
    headerRow.append(
        $("<th>").attr("data-y-col", "true")
                 .attr("contenteditable", "true")
                 .text(yHeader)
                 .css({ "background-color": "#f4b400", "color": "white" })
    );
    
    // 刪除欄
    headerRow.append(
        $("<th>").text("刪除")
                 .css({ "background-color": "#d32f2f", "color": "white" })
    );

    // 重建 bounds-row 並保留原有的值
    let boundsRow = $("#bounds-row");
    boundsRow.empty();
    
    xHeaders.forEach(h => {
        const savedBound = savedBounds[h] || { min: "", max: "" };
        let td = $("<td>").append(
            $("<div>").addClass("bounds-container").append(
                $("<label>").addClass("bound-label").text("上界"),
                $("<input>").attr("type", "number")
                           .attr("step", "0.01")
                           .addClass("bound-max")
                           .val(savedBound.max)
            )
        ).append(
            $("<div>").addClass("bounds-container").append(
                $("<label>").addClass("bound-label").text("下界"),
                $("<input>").attr("type", "number")
                           .attr("step", "0.01")
                           .addClass("bound-min")
                           .val(savedBound.min)
            )
        );
        boundsRow.append(td);
    });

    // y & 刪除 => 2格空白
    boundsRow.append("<td></td><td></td>");
}

  /* 增加行*/
  $("#add-row-btn").click(function(){
    let totalCols = $("#header-row th").length;
    let tr = $("<tr>").addClass("data-row");
    $("#header-row th").each(function(index){
      if(index===totalCols-1){
        tr.append($("<td>").append($("<button>").addClass("delete-row").text("X")));
      } else if(index===totalCols-2){
        tr.append($("<td>").append($("<input>").attr("type","number").attr("step","0.1")));
      } else {
        tr.append($("<td>").append($("<input>").attr("type","number").attr("step","0.1")));
      }
    });
    $("#bounds-row").before(tr);
  });

  /* 刪除行*/
  $(document).on("click", ".delete-row", function(e){
    // 彈出確認對話
    if (!confirm("確定要刪除此行嗎？")) {
        return; // 若使用者按「取消」，就中止
    }
    $(this).closest("tr").remove();
});  

  /* 增 X*/
  $(".adjust-btn[data-type='x'][data-action='add']").click(function(){
    let xHeaders = $("#header-row th[data-x-col='true']");
    let xCount = xHeaders.length;
    let newX = `x${xCount+1}`;

    let newHeader = $("<th>").attr("data-x-col","true").attr("contenteditable","true").text(newX);
    // 插在 y & 刪除前
    $("#header-row th:nth-last-child(2)").before(newHeader);

    $(".data-row").each(function(){
      $("<td>").append($("<input>").attr("type","number").attr("step","0.1"))
               .insertBefore($(this).find("td:nth-last-child(2)"));
    });
    $("<td>").append(
      $("<div>").addClass("bounds-container").append(
        $("<label>").addClass("bound-label").text("上界"),
        $("<input>").attr("type","number").attr("step","0.1").addClass("bound-max")
      )
    ).append(
      $("<div>").addClass("bounds-container").append(
        $("<label>").addClass("bound-label").text("下界"),
        $("<input>").attr("type","number").attr("step","0.1").addClass("bound-min")
      )
    ).insertBefore($("#bounds-row td:nth-last-child(2)"));
  });

  /* 減 X*/
  $(".adjust-btn[data-type='x'][data-action='remove']").click(function(){
    let xHeaders = $("#header-row th[data-x-col='true']");
    if(xHeaders.length>1){
      let lastXIndex = xHeaders.last().index();
      $("#header-row th").eq(lastXIndex).remove();
      $(".data-row").each(function(){
        $(this).find("td").eq(lastXIndex).remove();
      });
      $("#bounds-row td").eq(lastXIndex).remove();
    } else {
      alert("❌ 至少需要一個 X 變數！");
    }
  });

  /* 3. 儲存專案 => points.csv(含 X+Y), bounds.csv(只X)*/
  $("#save-project-btn").click(function(){
    let projectName = $("#project-name").val().trim();
    if(!projectName){
      alert("❌ 請輸入專案名稱！");
      return;
    }

    // 檢查同名
    $.ajax({
      type:"GET",
      url:"/project_exists?name="+encodeURIComponent(projectName),
      success:function(res){
        if(res.exists){
          if(!confirm(`專案「${projectName}」已存在，是否覆寫？`)){
            return;
          }
        }
        saveProjectToServer(projectName);
      },
      error:function(err){
        alert("❌ 無法檢查專案是否存在："+(err.responseJSON?.error||"未知錯誤"));
      }
    });
  });

  function saveProjectToServer(projectName){
    let pointsCsv=generatePointsCsv();
    let boundsCsv=generateBoundsCsv();

    let formData=new FormData();
    formData.append("project_name", projectName);
    formData.append("points_csv", pointsCsv);
    formData.append("bounds_csv", boundsCsv);

    $.ajax({
      type: "POST",
      url: "/save_project",
      data: formData,
      processData: false,
      contentType: false,
      success: function() {
        alert(`✅ 專案「${projectName}」已成功儲存！`);
      },
      error: function(err){
        alert("❌ 儲存失敗："+(err.responseJSON?.error||"未知錯誤"));
      }
    });
  }

  function generatePointsCsv(){
    let totalCols = $("#header-row th").length;
    let xThs = $("#header-row th[data-x-col='true']");
    let yIndex = totalCols - 2;

    // 標題
    let BOM = "\uFEFF";
    let xTitles = xThs.map((i,el)=>$(el).text().trim()).get();
    let yTitle = $("#header-row th").eq(yIndex).text().trim();
    let csvContent = BOM + xTitles.join(",") + "," + yTitle + "\n";

    // 逐行
    $(".data-row").each(function(){
      let rowData=[];
      xThs.each((_,th)=>{
        let colIndex=$(th).index();
        let val=$(this).find("td").eq(colIndex).find("input").val()||"";
        rowData.push(val);
      });
      let yVal=$(this).find("td").eq(yIndex).find("input").val()||"";
      rowData.push(yVal);
      csvContent += rowData.join(",") + "\n";
    });
    return csvContent;
  }

  function generateBoundsCsv(){
    let xThs = $("#header-row th[data-x-col='true']");
    if(xThs.length===0) return "";

    // 第一行 => X 標題
    let BOM = "\uFEFF";
    let headerTexts = xThs.map((i,el)=>$(el).text().trim()).get();
    let csvContent = BOM + headerTexts.join(",") + "\n";

    // 第二行 => 下界, 第三行 => 上界
    let lowers=[], uppers=[];
    xThs.each(function(){
      let colIndex = $(this).index();
      let lowerVal = $("#bounds-row td").eq(colIndex).find(".bound-min").val()||"";
      let upperVal = $("#bounds-row td").eq(colIndex).find(".bound-max").val()||"";
      lowers.push(lowerVal);
      uppers.push(upperVal);
    });
    csvContent += lowers.join(",") + "\n";
    csvContent += uppers.join(",") + "\n";
    return csvContent;
  }

   /* 4. 另存新檔 => 讓使用者挑位置下載當前表格 csv*/
  $("#export-csv-btn").click(async function(){
    try {
      let csvContent = generatePointsCsv();
      let suggestedName = "當前表格點位.csv";

      const handle = await window.showSaveFilePicker({
        suggestedName,
        types: [
          {
            description: "CSV Files",
            accept: { "text/csv": [".csv"] }
          }
        ]
      });
      const writable = await handle.createWritable();
      await writable.write(csvContent);
      await writable.close();

      alert("✅ 已另存新檔！");
    } catch(e){
      if(e.name!=="AbortError"){
        alert("❌ 另存新檔失敗："+ e.message);
      }
    }
  });

   /* 5. 生成 UD 樣本*/
  $("#generate-ud-btn").click(function () {
    let num_samples = parseInt($("#num-samples").val(), 10);
    if (isNaN(num_samples) || num_samples <= 0) {
      alert("請輸入有效的樣本數量！");
      return;
    }

    let xBounds = [];
    $("#bounds-row td").each(function () {
      let lower = $(this).find(".bound-min").val();
      let upper = $(this).find(".bound-max").val();
      if (lower !== undefined && upper !== undefined) {
        xBounds.push([parseFloat(lower), parseFloat(upper)]);
      }
    });

    if (xBounds.length === 0) {
      alert("請設定 X 變數的上下界！");
      return;
    }

    $.ajax({
      type: "POST",
      url: "/generate_samples",
      contentType: "application/json",
      data: JSON.stringify({ num_samples, bounds: xBounds }),
      success: function (res) {
        let samples = res.samples;
        $(".data-row").remove();
        samples.forEach(row => {
          let tr = $("<tr>").addClass("data-row");
          row.forEach(val => {
            tr.append(
              $("<td>").append($("<input>").attr("type", "number").attr("step", "0.1").val(val))
            );
          });
          // 補一欄 Y
          tr.append($("<td>").append($("<input>").attr("type","number").attr("step","0.1")));
          // 刪除
          tr.append($("<td>").append($("<button>").addClass("delete-row").text("X")));
          $("#bounds-row").before(tr);
        });
      },
      error: function (err) {
        alert("❌ 生成樣本失敗：" + (err.responseJSON?.error || "未知錯誤"));
      }
    });
  });

  /* 6. 生成 LHS 樣本*/
  $("#generate-lhs-btn").click(function () {
      let numSamples = parseInt($("#num-samples").val(), 10);
      if (isNaN(numSamples) || numSamples <= 0) {
        alert("請輸入有效的樣本數量！");
        return;
      }
    
      let bounds = [];
      $("#bounds-row td").each(function () {
        let lower = $(this).find(".bound-min").val();
        let upper = $(this).find(".bound-max").val();
        if (lower !== undefined && upper !== undefined) {
          bounds.push([parseFloat(lower), parseFloat(upper)]);
        }
      });
    
      if (bounds.length === 0) {
        alert("❌ 無有效上下界，請檢查輸入！");
        return;
      }
    
      // 對應後端 /generate_lhs
      $.ajax({
        type: "POST",
        url: "/generate_lhs",
        contentType: "application/json",
        data: JSON.stringify({
          num_samples: numSamples,
          bounds: bounds
        }),
        success: function (res) {
          // 後端若成功 -> res.samples
          fillTableWithSamples(res.samples);
        },
        error: function (err) {
          alert("❌ LHS 生成失敗：" + (err.responseJSON?.error || "未知錯誤"));
        }
      });
    });
    
    // 專門用來把 samples 填入表格
    function fillTableWithSamples(samples) {
      $(".data-row").remove();
      samples.forEach(row => {
        let tr = $("<tr>").addClass("data-row");
        row.forEach(val => {
          tr.append(
            $("<td>").append($("<input>").attr("type","number").attr("step","0.1").val(val))
          );
        });
        // 最後補一欄 Y
        tr.append($("<td>").append($("<input>").attr("type","number").attr("step","0.1")));
        tr.append($("<td>").append($("<button>").addClass("delete-row").text("X")));
        $("#bounds-row").before(tr);
      });
    }      

  /* 7. 上傳 CSV => 第一行: X...,Y*/
  $("#confirm-upload-btn").click(function () {
    let file = $("#file")[0].files[0];
    if (!file) {
      alert("❌ 請選擇 CSV 檔案！");
      return;
    }

    let reader = new FileReader();
    reader.onload = function (e) {
      let csvContent = e.target.result.trim();
      if (!csvContent) {
        alert("❌ CSV 檔案是空的！");
        return;
      }
      let rows = csvContent.split("\n").map(r => r.split(","));
      if (rows.length < 2) {
        alert("❌ CSV 至少需要標題列 + 數據列");
        return;
      }

      // 第一行 => 標題
      let headerRow = rows[0].map(c => c.trim());
      if (headerRow.length < 2) {
        alert("❌ CSV 需至少一個 X + 一個 Y 欄");
        return;
      }
      let xHeaders = headerRow.slice(0, headerRow.length - 1);
      let yHeader = headerRow[headerRow.length - 1];

      updateTableHeaders(xHeaders, yHeader);

      let dataRows = rows.slice(1);
      fillTableWithCSVData(dataRows);
    };
    reader.readAsText(file);
  });

  function fillTableWithCSVData(dataRows) {
    $(".data-row").remove();  // 只移除數據行
    
    dataRows.forEach(row => {
        // 若非全數字 => 跳過
        let allNumeric = row.every(cell => !isNaN(cell) && cell.trim() !== "");
        if (!allNumeric) return;

        let tr = $("<tr>").addClass("data-row");
        let lastIndex = row.length - 1;
        row.forEach((val, i) => {
            if (i < lastIndex) {
                tr.append($("<td>").append(
                    $("<input>").attr("type","number").attr("step","0.01").val(val.trim())
                ));
            } else {
                // y
                tr.append($("<td>").append(
                    $("<input>").attr("type","number").attr("step","0.01").val(val.trim())
                ));
            }
        });
        tr.append($("<td>").append($("<button>").addClass("delete-row").text("X")));
        $("#bounds-row").before(tr);
    });
    console.log("✅ 已將 CSV 數據填入表格");
}

$("#confirm-upload-btn").click(function () {
    let file = $("#file")[0].files[0];
    if (!file) {
        alert("❌ 請選擇 CSV 檔案！");
        return;
    }

    let reader = new FileReader();
    reader.onload = function (e) {
        let csvContent = e.target.result.trim();
        if (!csvContent) {
            alert("❌ CSV 檔案是空的！");
            return;
        }
        let rows = csvContent.split("\n").map(r => r.split(","));
        if (rows.length < 2) {
            alert("❌ CSV 至少需要標題列 + 數據列");
            return;
        }

        // 第一行 => 標題
        let headerRow = rows[0].map(c => c.trim());
        if (headerRow.length < 2) {
            alert("❌ CSV 需至少一個 X + 一個 Y 欄");
            return;
        }
        let xHeaders = headerRow.slice(0, headerRow.length - 1);
        let yHeader = headerRow[headerRow.length - 1];

        // 保存當前的上下界值
        const savedBounds = {};
        $("#header-row th[data-x-col='true']").each(function(index) {
            const header = $(this).text().trim();
            const td = $("#bounds-row td").eq(index);
            if (td.length) {
                savedBounds[header] = {
                    min: td.find(".bound-min").val(),
                    max: td.find(".bound-max").val()
                };
            }
        });

        // 更新表頭
        updateTableHeaders(xHeaders, yHeader);

        // 恢復已有的上下界值
        xHeaders.forEach((header, index) => {
            if (savedBounds[header]) {
                const td = $("#bounds-row td").eq(index);
                td.find(".bound-min").val(savedBounds[header].min);
                td.find(".bound-max").val(savedBounds[header].max);
            }
        });

        let dataRows = rows.slice(1);
        fillTableWithCSVData(dataRows);
    };
    reader.readAsText(file);
});

/* 8. 模型類型切換時的處理*/
$("#train-model-type").on("change", function() {
  const modelType = $(this).val();
  if (modelType === "MLP") {
      $("#mlp-params").show();
      $("#rsm-params").hide();
  } else {
      $("#mlp-params").hide();
      $("#rsm-params").show();
  }
});

$("#random-state-type").on("change", function() {
  if ($(this).val() === "fixed") {
      $("#random-state-container").show();
  } else {
      $("#random-state-container").hide();
  }
});

/* 8. 訓練按鈕事件*/
$("#train-model-btn").on("click", function() {
  // 從表格中獲取數據
  const tableData = [];
  const headers = [];
  
  // 獲取表頭
  $("#header-row th").each(function() {
      if($(this).attr("data-x-col") || $(this).attr("data-y-col")) {
          headers.push($(this).text().trim());
      }
  });

  // 獲取數據行
  $(".data-row").each(function() {
      const rowData = [];
      $(this).find("input[type='number']").each(function() {
          const val = $(this).val();
          if(val !== "") {
              rowData.push(parseFloat(val));
          }
      });
      if(rowData.length === headers.length) {  // 確保數據完整
          tableData.push(rowData);
      }
  });

  if(tableData.length === 0) {
      alert("請先輸入數據！");
      return;
  }

  let modelType = $("#train-model-type").val();
  let payload = {
      model_type: modelType,
      data: tableData,
      project_name: $("#project-name").val().trim() || null
  };

  // 根據模型類型獲取相應的 R² 閾值
  let minR2;
  if (modelType === "MLP") {
    minR2 = parseFloat($("#min-r2").val()) || 0.8;
    
    // 獲取 MLP 特定參數
    let hiddenLayersStr = $("#mlp-structure").val().trim();
    if (!hiddenLayersStr) {
        alert("請輸入隱藏層結構！");
        return;
    }
    try {
        let layers = hiddenLayersStr.split(',').map(x => {
            let num = parseInt(x.trim());
            if (isNaN(num) || num <= 0) throw new Error('無效的數值');
            return num;
        });
        payload.hidden_layers = layers;
        payload.activation = $("#activation").val();
        payload.max_iter = parseInt($("#max-iter").val());
        payload.learning_rate = parseFloat($("#learning-rate").val());
        payload.batch_size = parseInt($("#batch-size").val());

        // 只有在選擇固定種子時才設置 random_state
        if ($("#random-state-type").val() === "fixed") {
            const seedValue = parseInt($("#random-state").val());
            if (isNaN(seedValue) || seedValue < 0 || seedValue >= Math.pow(2, 32)) {
                alert("種子值必須在 0 到 " + (Math.pow(2, 32) - 1) + " 之間");
                return;
            }
            payload.random_state = seedValue;
        }
        // 選擇隨機時
        else {
          payload.random_state = NaN;
        }

    } catch (e) {
        alert("隱藏層結構格式錯誤！請用逗號分隔的正整數，例如：3,27,5");
        return;
    }
  }else {  // RSM
    minR2 = parseFloat($("#rsm-min-r2").val()) || 0.8;
  }

  // 驗證 R² 閾值
  if (isNaN(minR2) || minR2 < 0 || minR2 > 1) {
      alert("請輸入有效的 R² 閾值（0-1之間）！");
      return;
  }

  // 將 R² 閾值加入 payload
  payload.min_r2 = minR2;

  $.ajax({
      type: "POST",
      url: "/train_model",
      contentType: "application/json",
      data: JSON.stringify(payload),
      success: function(res) {
          // 更新 R² 值顯示
          if (res.r2 !== undefined) {
              const r2Value = res.r2.toFixed(4);
              $("#r2-display").show();
              $("#r2-value")
                  .text(r2Value)
                  .css("color", res.r2 >= minR2 ? "#3b7d97" : "red");

              // 如果 R² 太低，顯示警告
              if (res.r2 < minR2) {
                  $("#r2-warning").show();
              } else {
                  $("#r2-warning").hide();
              }
          }

          alert("✅ 訓練成功：" + res.message);
      },
      error: function(err) {
          alert("❌ 訓練失敗：" + (err.responseJSON?.error || "未知錯誤"));
      }
  });
});

/* 9. 點擊預測新點位按鈕的處理*/
$("#predict-new-points-btn").click(function() {
  const numNewPoints = parseInt($("#new-points-count").val());
  if (isNaN(numNewPoints) || numNewPoints <= 0) {
      alert("請輸入有效的點位數量！");
      return;
  }

  // 收集當前表格數據
  const currentData = [];
  $(".data-row").each(function() {
      const rowData = [];
      $(this).find("input[type='number']").each(function() {
          rowData.push(parseFloat($(this).val()) || 0);
      });
      currentData.push(rowData);
  });

  // 收集邊界值
  const bounds = [];
  $("#bounds-row td").each(function() {
      const min = $(this).find(".bound-min").val();
      const max = $(this).find(".bound-max").val();
      if (min !== undefined && max !== undefined) {
          bounds.push([parseFloat(min), parseFloat(max)]);
      }
  });

  $.ajax({
      type: "POST",
      url: "/generate_new_points",
      contentType: "application/json",
      data: JSON.stringify({
          current_data: currentData,
          bounds: bounds,
          num_points: numNewPoints
      }),
      success: function(res) {
          handleNewPoints(res);
      },
      error: function(err) {
          alert("❌ 生成點位失敗：" + (err.responseJSON?.error || "未知錯誤"));
      }
  });
});

function handleNewPoints(res) {
  // 先移除現有的結果
  $(".prediction-results").remove();
  
  // 獲取 X 變數的標題
  const xHeaders = $("#header-row th[data-x-col='true']")
      .map(function() { return $(this).text(); })
      .get();
  
  // 創建新的結果表格
  const resultsHtml = `
      <div class="white-card prediction-results">
          <table class="results-table">
              <thead>
                  <tr>
                      ${xHeaders.map(h => `<th>${h}</th>`).join('')}
                  </tr>
              </thead>
              <tbody>
                  ${res.new_points.map(point => `
                      <tr class="point-row">
                          ${point.map(val => `<td>${parseFloat(val).toFixed(2)}</td>`).join('')}
                      </tr>
                  `).join('')}
                  <tr class="bounds-row">
                      ${res.new_bounds.map(bound => `
                          <td>
                              <div>上界: ${parseFloat(bound[1]).toFixed(2)}</div>
                              <div>下界: ${parseFloat(bound[0]).toFixed(2)}</div>
                          </td>
                      `).join('')}
                  </tr>
              </tbody>
          </table>
          <button class="add-to-main-table">加入主表格</button>
      </div>
  `;

  // 將結果添加到頁面
  $(".model-training-container").after(resultsHtml);
}

/* 10. 加入主表格*/
$(document).on('click', '.add-to-main-table', function() {
  // 獲取所有預測點位
  $(".results-table tbody tr.point-row").each(function() {
      const newPoint = [];
      $(this).find('td').each(function() {
          newPoint.push(parseFloat($(this).text()).toFixed(2));
      });
      
      // 為每個點位創建新行
      const tr = $("<tr>").addClass("data-row");
      
      // 添加 X 值
      newPoint.forEach(val => {
          tr.append($("<td>").append(
              $("<input>")
                  .attr("type", "number")
                  .attr("step", "0.01")
                  .val(val)
          ));
      });
      
      // 添加空的 Y 值欄位
      tr.append($("<td>").append(
          $("<input>").attr("type", "number").attr("step", "0.01")
      ));
      
      // 添加刪除按鈕
      tr.append($("<td>").append(
          $("<button>").addClass("delete-row").text("X")
      ));
      
      // 將新行添加到主表格
      $("#bounds-row").before(tr);
  });
  
  // 更新邊界值
  $(".results-table tbody tr.bounds-row td").each(function(index) {
      const upper = parseFloat($(this).find('div:first').text().split(': ')[1]).toFixed(2);
      const lower = parseFloat($(this).find('div:last').text().split(': ')[1]).toFixed(2);
      
      $("#bounds-row td").eq(index).find('.bound-max').val(upper);
      $("#bounds-row td").eq(index).find('.bound-min').val(lower);
  });
  
  // 移除預測結果
  $(".prediction-results").remove();
});

}); // document.ready 結尾