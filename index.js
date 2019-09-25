const modelImgSize = 128;

// https://stackoverflow.com/questions/48969495/in-javascript-how-do-i-should-i-use-async-await-with-xmlhttprequest
function makeRequest(method, url) {
    return new Promise(function (resolve, reject) {
        let xhr = new XMLHttpRequest();
        xhr.open(method, url);
        xhr.responseType = "arraybuffer";
        xhr.onload = function () {
            if (this.status >= 200 && this.status < 300) {
                resolve(xhr.response);
            } else {
                reject({
                    status: this.status,
                    statusText: xhr.statusText
                });
            }
        };
        xhr.onerror = function () {
            reject({
                status: this.status,
                statusText: xhr.statusText
            });
        };
        xhr.send();
    });
}

var modelData = "./g.dn4.onnx";


async function processImgPortion(imgdata, xo, yo)
{
    var raw_to_process = new Float32Array(3 * modelImgSize * modelImgSize).fill(0.0);
    var to_process = ndarray(raw_to_process, [3, modelImgSize, modelImgSize]);

    var top_i = Math.min(modelImgSize, imgdata.shape[1] - xo) - 1;
    var top_y = Math.min(modelImgSize, imgdata.shape[0] - yo) - 1;

    var x = 0;
    var y = 0;
    for(var i = 0; i < modelImgSize; ++i) 
    {
        for(var j = 0; j < modelImgSize; ++j) 
        {
            x = Math.min(i, top_i);
            y = Math.min(j, top_y);
            to_process.set(0, i, j, imgdata.get(yo + y, xo + x, 0) / 255.0);
            to_process.set(1, i, j, imgdata.get(yo + y, xo + x, 1) / 255.0);
            to_process.set(2, i, j, imgdata.get(yo + y, xo + x, 2) / 255.0);
        }
    }

    const session = new onnx.InferenceSession({ backendHint: 'webgl' });
    if(modelData.startsWith("./g."))
    {
        modelData = await makeRequest("GET", modelData);   
    }
    await session.loadModel(modelData);
    var inputTensor = new onnx.Tensor(to_process.data, 'float32', [3, modelImgSize, modelImgSize]);
    var outputMap = await session.run([inputTensor]);
    var outputData = ndarray(outputMap.values().next().value.data, [3, modelImgSize, modelImgSize]);

    for(var i = 0; i <= top_i; ++i) 
    {
        for(var j = 0; j <= top_y; ++j) 
        {

            imgdata.set(yo + j, xo + i, 0, outputData.get(0, i, j) * 255);
            imgdata.set(yo + j, xo + i, 1, outputData.get(1, i, j) * 255);
            imgdata.set(yo + j, xo + i, 2, outputData.get(2, i, j) * 255);
        }
    }
    return imgdata;
}

function generateGuid()
{
    return Math.random().toString(36).substring(2, 15) + Math.random().toString(36).substring(2, 15);
}

function createImgDisplay()
{
    var maintag = document.getElementById("main");
    var article = document.createElement("article");
    var imgid = "img-" + generateGuid();
    var imgtag = document.createElement("img");
    imgtag.src = "./loading.gif";
    
    var rNCCtag = document.createElement("input");
    rNCCtag.type = "button";
    rNCCtag.value = "Correct Compression";
    rNCCtag.onclick = function() {runNeuralCompressionCorrection(imgid, 1);}

    var enhancetag = document.createElement("input");
    enhancetag.type = "button";
    enhancetag.value = "\"Enhance\"";
    enhancetag.onclick = function() {runNeuralCompressionCorrection(imgid, 2);}

    imgtag.id = imgid;

    article.appendChild(imgtag);
    article.appendChild(document.createElement("br"));
    article.appendChild(rNCCtag);
    article.appendChild(enhancetag);
    maintag.appendChild(article);

    return imgid;
}

async function runNeuralCompressionCorrection(image_id, scale) 
{
    var imgid = createImgDisplay();
    var imgtag = document.getElementById(imgid);

    var sourceimage = document.getElementById(image_id);
    sourceimage.crossOrigin = "Anonymous";
    var canvas = document.createElement('canvas');
    canvas.height = canvas.width = 0;
    var imgwidth = sourceimage.offsetWidth * scale;
    var imgheight = sourceimage.offsetHeight * scale;

    canvas.height = imgheight;
    canvas.width = imgwidth;
    context = canvas.getContext('2d');
    context.drawImage(sourceimage, 0, 0, imgwidth, imgheight);
    const imageData = context.getImageData(0, 0, imgwidth, imgheight);

    var dataFromImage = ndarray(new Uint8ClampedArray(imageData.data), [imgheight, imgwidth, 4]);
    var imgstep = modelImgSize - 8;

    for(var xo = 0; xo < imgwidth; xo += imgstep)
    {
        for(var yo = 0; yo < imgheight; yo += imgstep)
        {
            await processImgPortion(dataFromImage, xo, yo);
        }
    }

    var idata = context.createImageData(imgwidth, imgheight);
    idata.data.set(dataFromImage.data);
    context.putImageData(idata, 0, 0);

    imgtag.src = canvas.toDataURL();
}

function onFileSelected(event)
{
    var selectedFile = event.target.files[0];
    var reader = new FileReader();
    
    var imgid = createImgDisplay();
    var imgtag = document.getElementById(imgid);

    imgtag.title = selectedFile.name;
  
    reader.onload = function(event) {
      imgtag.src = event.target.result;
    };
  
    reader.readAsDataURL(selectedFile);
}
