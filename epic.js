//sowwy if my code is hard to read :3

const models = {
    PEANUT: {
        name: "Peanut",
        modelFolder: "peanut",
        headshotFolder: "peanut",
        useEncoder: true
    },

    ALL_CHARS: {
        name: "All Characters",
        modelFolder: "all_chars",
        useEncoder: false
    },
    
    FOX: {
        name: "Fox",
        modelFolder: "fox",
        headshotFolder: "fox",
        useEncoder: true
    },
    KING:{
        name: "King",
        modelFolder: "King",
        headshotFolder: "King",
        useEncoder: true
    }
}

const STATUS = {
    SUCCESS: "rgb(155, 255, 155)",
    FAIL: "#f00",
    INFO: "#fff"
}

const SCALE = 2;
const WIDTH = SCALE*128;
const HEIGHT = SCALE*128;

var totalPredictions = 0;
var startTime = 0;
var lastSessionAPT = 150;

function getAvgPredictTime(){
    if(totalPredictions > 0 && startTime>0){
        return (Date.now()-startTime)/totalPredictions
    }
    return  lastSessionAPT;
}

function startTiming(){
    totalPredictions = 0;
    startTime = Date.now();
}

function finishTiming(){
    if(totalPredictions>0){
        lastSessionAPT = (Date.now()-startTime)/totalPredictions;
    }else
        lastSessionAPT = lastSessionAPT;
    startTime = 0;
}

class Model{
    constructor(generatorModel, metadata, encoderModel, modelDescriptor, headshotIndex=null){
        this.generatorModel = generatorModel;
        this.inputParamSize = metadata[0];
        this.outputImageSize = metadata[1];
        this.encoderModel = encoderModel;
        this.usesEncoder = modelDescriptor.useEncoder;
        this.name = modelDescriptor.name;
        this.headshotIndex = headshotIndex;
        this.headshotFolder = modelDescriptor.headshotFolder;
    }

    async predictFromParams(params){
        console.assert(params.length == this.inputParamSize);
        setStatus("Loading parameters...", STATUS.INFO);
        let tensor =  this.generatorModel.predict(tf.tensor([params]));
        tensor = tensor.mul(255.0/2)
        tensor = tensor.add(255/2)
        setStatus("Predicting...", STATUS.INFO);
        let returner = (await tensor.array())[0]
        setStatus("Done!", STATUS.SUCCESS)
        totalPredictions++;
        return returner;
    }

    async getParamsFromImage(imageArray){
        console.assert(this.usesEncoder && imageArray.length == 3 * this.outputImageSize * this.outputImageSize);
        let imageTensor = tf.tensor(imageArray);
        imageTensor = imageTensor.reshape([1,this.outputImageSize, this.outputImageSize, 3]);
        imageTensor = imageTensor.transpose([0,3,1,2]);
        imageTensor = imageTensor.sub(255/2).div(255/2)
        let paramTensor = this.encoderModel.predict(imageTensor);
        setStatus("Loading Image Params...", STATUS.INFO);
        let params = (await paramTensor.array())[0];
        return params;
    }

    getRandomHeadshots(min=16){
        let shuffledArray = this.headshotIndex.slice(), i=this.headshotIndex.length, temp, index;
        while(i-- > min){
            index = Math.floor((i+1)*Math.random())
            temp = shuffledArray[index];
            shuffledArray[index] = shuffledArray[i]
            shuffledArray[i] = temp;
        }
        return shuffledArray.slice(0,min).map(temp=>`headshots/${this.headshotFolder}/${temp}`);
    }

}

async function loadKerasModel(modelDescriptor){
    setStatus(`Loading ${modelDescriptor.name} model`, STATUS.INFO);
    document.getElementById("modelSelect").disabled = true;
    for(let button of Object.values(BUTTONS))
        setEnabledState(button, false);
    const generatorModel = await tf.loadLayersModel(`models/model_${modelDescriptor.modelFolder}/model.json`)
    const metadata = await ajaxJSON(`models/model_${modelDescriptor.modelFolder}/metadata.json`)
    if(modelDescriptor.useEncoder){
        const encoderModel = await tf.loadLayersModel(`models/model_${modelDescriptor.modelFolder}_enc/model.json`);
        const headshotIndex = (await ajax(`headshots/${modelDescriptor.headshotFolder}/index.txt`)).split('\n').filter(x=>x.length>0);
        model =  new Model(generatorModel, metadata, encoderModel, modelDescriptor, headshotIndex);
    }else
        model = new Model(generatorModel, metadata, null, modelDescriptor);
    generateRandomParams();
    createParamSliders();
    setDefaultEnabledState();
    document.getElementById("modelSelect").disabled = false;
}

async function ajax(url){
    return new Promise((resolve, reject)=>{
        var xhttp = new XMLHttpRequest();
        xhttp.onreadystatechange = function() {
            if (this.readyState == 4 && this.status == 200) {
                if(this.status == 200){
                    resolve(this.responseText);
                }else{
                    reject("Failed to ajax to "+url);
                }
            }
        };
        xhttp.open("GET", url, true);
        xhttp.send();
    });
}

async function ajaxJSON(url){
    return JSON.parse((await ajax(url)));
}

function setStatus(status, color){
    let statusElement = document.getElementById("status");
    statusElement.innerText = status;
    statusElement.style['color'] = color;
    if(color == STATUS.FAIL)
        console.error(status);
    window.a = status;
}


var model;
var outputArray;
var params = [];
var paramSliders = [];


var isSliding = false;
const MILLIS_PER_ITER = 600;

const BUTTONS = {
    randomButton: null,
    loadButton: null,
    sliderButton: null,
    interpolateButton: null,
    saveButton: null
};

function setEnabledState(button, enabled){
    if(enabled){
        button.classList.remove("disabled");
        if(button.tagName == "DIV"){
            button.querySelector("input").disabled = false;
        }else
            button.disabled = false;
    }else{
        button.classList.add("disabled");
        if(button.tagName == "DIV"){
            button.querySelector("input").disabled = true;
        }else
            button.disabled = true;
    }
}

function setActivatedState(button, activated){
    if(activated)
        button.classList.add("activated");
    else
        button.classList.remove("activated");
}

async function wait(millis){
    return new Promise((resolve, reject)=>{
        setTimeout(resolve, millis);
    });
}

function setDefaultEnabledState(){
    for(let button of Object.values(BUTTONS)){
        if(!model.usesEncoder && (button.id=="loadButton" || button.id=="interpolateButton"))
            setEnabledState(button, false);
        else
            setEnabledState(button, true);
    }
}


function* slideGenerator(){

    startTiming();
    let params_0 = params.slice();
    while(isSliding){
        let params_1 = tf.randomNormal([model.inputParamSize],0,1).arraySync();
        let numIterations = Math.round(MILLIS_PER_ITER/getAvgPredictTime());
        if(numIterations < 8)
            numIterations = 8;
        
        for(let i=0;i<numIterations&&isSliding;i++){
            let p1_mult = (i+1)/numIterations;
            let p0_mult = 1 - p1_mult;
            let slideParams = [];
            
            for(let j=0;j<params.length;j++){
                slideParams[j] = (params_0[j]*p0_mult+params_1[j]*p1_mult);
            }
    
            yield updateImage(slideParams).then(()=>updateSliders());
        }

        params_0 = params_1;
    }
    finishTiming();
    
    return;
}

const MAX_PROMISES = 3;
function doTheSlide(){
    setActivatedState(BUTTONS.sliderButton, true);
    for(let button of Object.values(BUTTONS)){
        if(button != BUTTONS.sliderButton)
            setEnabledState(button, false);
    }
    document.getElementById("modelSelect").disabled = true;
    isSliding = true;

    let generator = slideGenerator();
    
    let promises = Array.apply(null, Array(MAX_PROMISES)).map(()=>generator.next().value.then(promiseFulfilled).catch(error=>setStatus(error, STATUS.FAIL)));

    function promiseFulfilled(){
        promises.shift();
        let next = generator.next();
        if(!next.done){
            promises.push(generator.next().value.then(promiseFulfilled).catch(error=>setStatus(error, STATUS.FAIL)))
        }else if(promises.length == 0){
            setDefaultEnabledState();
            document.getElementById("modelSelect").disabled = false;
            setActivatedState(BUTTONS.sliderButton, false);
        }
    }
}


function toggleSlider(){
    if(isSliding){
        isSliding = false;
        setEnabledState(BUTTONS.sliderButton, false);
        return;
    }

    doTheSlide();
}

function getRandomParams(){
    return tf.randomNormal([model.inputParamSize],0,1).arraySync();
}

function generateRandomParams(){
    params = tf.randomNormal([model.inputParamSize],0,1).arraySync();
}

async function loadImageParams(path){
    let imageElement = document.createElement("img");
    imageElement.src = path;
    imageElement.crossOrigin = "anonymous";
    await new Promise((resolve, reject)=>{
        imageElement.addEventListener("load",resolve);
    })
    let canvas = document.createElement("canvas");
    canvas.width = model.outputImageSize;
    canvas.height = model.outputImageSize;
    canvas.style = "display:none";
    document.body.appendChild(canvas);
    await new Promise((resolve, reject)=>{
        canvas.addEventListener("load",resolve);
        setTimeout(resolve, 1);
    });
    

    let context = canvas.getContext("2d");
    context.drawImage(imageElement, 0,0,model.outputImageSize, model.outputImageSize);
    let imageData = context.getImageData(0,0,canvas.width, canvas.height).data;

    let pixelArray = [];
    let numPixels = model.outputImageSize*model.outputImageSize;
    for(let i=0;i<numPixels;i++){
        pixelArray[i*3+2] = imageData[i*4];
        pixelArray[i*3+1] = imageData[i*4+1];
        pixelArray[i*3+0] = imageData[i*4+2];
    }

    document.body.removeChild(canvas);

    return await model.getParamsFromImage(pixelArray);
}


function updateSliders(){
    for(let i=0;i<paramSliders.length;i++){
        paramSliders[i].value = ""+params[i];
    }
}

function sliderInput(event){
    if(isSliding)
        return;
    if(document.getElementById("activeChangeBox").checked){
        params[parseInt(event.target.id.substring(3))] = parseFloat(event.target.value);
        updateImage().catch(error=>setStatus(error, STATUS.FAIL));
    }
}

function sliderChanged(event){
    if(isSliding)
        return;
    params[parseInt(event.target.id.substring(3))] = parseFloat(event.target.value);
    updateImage().catch(error=>setStatus(error, STATUS.FAIL));
}

async function updateImage(params_=params){
    outputArray = await model.predictFromParams(params_);
    params = params_;
}

function createParamSliders(){
    let parameterDiv = document.getElementById("parameters");
    paramSliders = [];
    while(parameterDiv.firstChild)
        parameterDiv.removeChild(parameterDiv.firstChild);
    
    for(let i=0;i<params.length;i++){
        let slider = document.createElement("input");
        slider.type = "range";
        slider.min = "-3";
        slider.max = "3";
        slider.step = "0.5";
        slider.value = params[i].toFixed(1);
        slider.id = "sl_"+i;
        slider.addEventListener("change", sliderChanged);
        slider.addEventListener("input", sliderInput);
        paramSliders.push(slider);
        parameterDiv.append(slider);
    }
}

let modelMapping = {};
Object.values(models).forEach(modelDescriptor=>{
    modelMapping[modelDescriptor.modelFolder] = modelDescriptor;
});

var imgs = [];

function setup(){
    let canvas = createCanvas(WIDTH, HEIGHT);
    canvas.parent("container");

    for(let buttonID of Object.keys(BUTTONS)){
        BUTTONS[buttonID] = document.getElementById(buttonID);
    }

    let modelSelect = document.getElementById("modelSelect");
    Object.values(models).forEach(modelDescriptor=>{
        let modelOption = document.createElement("option");
        modelOption.value = modelDescriptor.modelFolder;
        modelOption.innerText = modelDescriptor.name;
        modelSelect.appendChild(modelOption);
    });

    modelSelect.addEventListener("change", function(event){
        let asnc = async function(){
            if(modelSelect.value && modelSelect.value != model.name){
                await loadKerasModel(modelMapping[modelSelect.value]);
                await updateImage();
            }
        }

        asnc().catch(error=>setStatus(error, STATUS.FAIL))
    });

    loadKerasModel(models.PEANUT).then(()=>{
        return model.predictFromParams(params);
    }).then(a=>outputArray=a).catch(error=>{
        setStatus(error, STATUS.FAIL);
    });

    let table = document.getElementById("imageTable");
    for(let i=0;i<4;i++){
        let row = document.createElement("tr");
        for(let j=0;j<5;j++){
            let cell = document.createElement("td");
            let img = document.createElement("img");
            img.crossOrigin = "anonymous";
            img.id = "s_img_"+(i*4+j);
            imgs.push(img);
            cell.appendChild(img);
            row.appendChild(cell);
        }
        table.appendChild(row);
    }

}



let firstDraw = true;
let frameCount = 0;
function draw(){
    if(firstDraw){
        background(255);
        fill(0);
        textAlign(CENTER, CENTER);
        textSize(15);
        text("Loading ...", WIDTH/2, HEIGHT/2);
        firstDraw = false;
    }
    if(outputArray){
        loadPixels();
        for(let x=0;x<WIDTH;x++){
            for(let y=0;y<HEIGHT;y++){
                let index = (y*HEIGHT + x)*4;
                let outputX = Math.floor(x/SCALE);
                let outputY = Math.floor(y/SCALE);
                pixels[index] = outputArray[2][outputY][outputX];
                pixels[index+1] = outputArray[1][outputY][outputX];
                pixels[index+2] = outputArray[0][outputY][outputX];
                pixels[index+3] = 255;
            }
        }
        updatePixels();

        if(outputArray.predictionStartTime){
            totalPredictionTime += (Date.now()-outputArray.predictionStartTime);
            totalPredictions++;
            predictionStartTime = 0;
        }
        outputArray = null;

    }

    if(frameCount%10==0){
        document.getElementById("fps").innerText = "FPS: "+Math.round(frameRate());
        if(isSliding)
            document.getElementById("pps").innerText = "PPS: "+Math.round(1000/getAvgPredictTime());
        else
            document.getElementById("pps").innerText = "";
    }
    frameCount++;
}

function generate(){
    generateRandomParams();
    updateSliders();
    updateImage().catch(error=>{
        setStatus(error, STATUS.FAIL);
    });
}

function loadFile(event){
    loadImageParams(URL.createObjectURL(event.target.files[0])).then(par=>{
        params = par;
        updateSliders();
        return updateImage();
    }).catch(error=>{
        setStatus(error, STATUS.FAIL);
    });
}

function saveImg(){
    save(model.name+"_"+Math.ceil(Math.random()*100000))
}

var isInterpolating = false;
var interpolationTaskQueue = [];
var interpolationParams = [0,0];

async function* interpolationEventListener(){
    let eventQueue = [];
    let res = null;
    interpolationEventListener.do = function(promise){
        eventQueue.push(promise);
        if(res != null)
            res();
    }
    while(isInterpolating){
        let promise = new Promise((resolve, reject)=>{
            res = resolve;
        });
        await promise;
        while(eventQueue.length > 0){
            let promise = eventQueue.shift();
            yield await promise;
        }
    }
    return 
}

async function interpolateAsync(){
    isInterpolating = true;
    interpolationParams = [getRandomParams(), getRandomParams()];

    BUTTONS.interpolateButton.classList.add("activated");
    for(let button of Object.values(BUTTONS)){
        if(button != BUTTONS.interpolateButton && button != BUTTONS.saveButton)
            setEnabledState(button, false);
    }
    document.getElementById("modelSelect").disabled = true;
    document.getElementById("parametersSection").style.display = "none";
    document.getElementById("interpolationSection").style.display = "block";

    let e = null;
    let eventListener = interpolationEventListener();
    for await(e of eventListener)
        await e;
}

function interpolate(){
    if(isInterpolating){
        isInterpolating = false;
        interpolationEventListener.do(0);
        return;
    }
    interpolateAsync().catch(error=>{
        setStatus(error, STATUS.FAIL);
    }).finally(()=>{
        document.getElementById("modelSelect").disabled = false;
        setDefaultEnabledState();
        document.getElementById("parametersSection").style.display = "block";
        document.getElementById("interpolationSection").style.display = "none";
        BUTTONS.interpolateButton.classList.remove("activated");
        document.getElementById("intImg0").src = "";
        document.getElementById("intImg1").src = "";
    });

    interpolationEventListener.do(
        Promise.all([
            setIntImage(0, model.getRandomHeadshots(1)[0]),
            setIntImage(1, model.getRandomHeadshots(1)[0])
        ])
    );
}


async function updateInterpolationImage(){
    let prop = parseFloat(document.getElementById("intSlider").value);
    let interpolatedParams = []
    for(let i=0;i<interpolationParams[0].length;i++)
        interpolatedParams[i] = interpolationParams[0][i]*(1-prop) + interpolationParams[1][i]*prop;
    params = interpolatedParams;
    updateSliders();

    await updateImage();
}

async function setIntImage(index, url){
    document.getElementById("intImg"+index).src = url;
    return loadImageParams(url).then(params_=>{
        interpolationParams[index]=params_;
        updateInterpolationImage();
    });
}

function selectInt(event){
    let endpointIndex = parseInt(event.target.id.charAt(9));
    interpolationEventListener.do(
        selectHeadshot().then(url => setIntImage(endpointIndex, url))
    );
}

let changeCount = 0;

function sliderValueChange(){
    if(isInterpolating){
        interpolationEventListener.do(updateInterpolationImage());
    }
}

function sliderValueInput(){
    if(isInterpolating && changeCount < MAX_PROMISES){
        changeCount++;
        interpolationEventListener.do(updateInterpolationImage().then(()=>changeCount--));
    }
}

function setOverlayState(enabled){
    if(enabled){
        document.getElementById("overlay").style.display = "flex";
    }else{
        document.getElementById("overlay").style.display = "none";
    }
}

async function selectHeadshot(){
    setOverlayState(true);

    return await new Promise((resolve, reject)=>{
        function imgSelected(event){
            let src = event.target.src;
            for(let img of imgs)
                img.src = "";
            setOverlayState(false);
            resolve(src);
        }

        function setRandomFaces(){
            let headshotSrcs = model.getRandomHeadshots(imgs.length);
            for(let i=0;i<headshotSrcs.length;i++){
                imgs[i].src = headshotSrcs[i];
                imgs[i].onclick = imgSelected;
            }
        }

        document.getElementById("refresh").onclick = setRandomFaces;

        setRandomFaces();

    });


}