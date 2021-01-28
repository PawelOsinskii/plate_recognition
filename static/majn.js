
function httpGet()
{
    const xmlHttp = new XMLHttpRequest();
    xmlHttp.open( "GET", '/get_plate', false ); // false for synchronous request

    xmlHttp.send( null );
    console.log(xmlHttp.responseText);
    document.getElementById("result").innerHTML = xmlHttp.responseText ;
    const img = document.createElement("img");

    img.src = "C:\\Users\\posinski\\PycharmProjects\\plate_recognition\\images_to_algorithm\\cropped.jpg";
    const src = document.getElementById("x");

    src.appendChild(img);
}

function showImage(){

}
