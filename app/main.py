from flask import send_from_directory
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
from flask import render_template
from url_utils import get_base_url
import os
import torch

# setup the webserver
# port may need to be changed if there are multiple flask servers running on same server
port = 12345
base_url = get_base_url(port)

# if the base url is not empty, then the server is running in development, and we need to specify the static folder so that the static files are served
if base_url == '/':
    app = Flask(__name__)
else:
    app = Flask(__name__, static_url_path=base_url+'static')

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024
selected_conversion = "USD"

torch.hub._validate_not_a_forked_repo=lambda a,b,c: True

model = torch.hub.load("ultralytics/yolov5", "custom", path = 'best2.pt', force_reload=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    
def conversion_logic(labels, image_path):
    
    from PIL import Image, ExifTags
    from PIL.ExifTags import TAGS, GPSTAGS
    from geopy.geocoders import Nominatim
    from babel.numbers import get_territory_currencies
    from forex_python.converter import CurrencyRates
    from unidecode import unidecode
    from countryinfo import CountryInfo
    import easyocr
    from googletrans import Translator, constants
    import re
    from word2number import w2n
    from currency_converter import CurrencyConverter

    class_label = labels # THIS CAN BE 'menu', 'price tag', OR 'currency'
#     preference = selected_conversion
    preference = "USD"

    # GPS EXIF DATA EXTRACTION
    image = Image.open(image_path)
#     exif_data = {}
#     info = image._getexif()
#     if info:
#         for tag, value in info.items():
#             decoded = TAGS.get(tag, tag)
#             if decoded == "GPSInfo":
#                 gps_data = {}
#                 for gps_tag in value:
#                     sub_decoded = GPSTAGS.get(gps_tag, gps_tag)
#                     gps_data[sub_decoded] = value[gps_tag]
#                 exif_data[decoded] = gps_data
#             else:
#                 exif_data[decoded] = value
#     if exif_data['GPSInfo']['GPSLongitudeRef'] == 'W': longitude = -exif_data['GPSInfo']['GPSLongitude'][0]
#     else: longitude = exif_data['GPSInfo']['GPSLongitude'][0]
#     if exif_data['GPSInfo']['GPSLatitudeRef'] == 'S': latitude = -exif_data['GPSInfo']['GPSLatitude'][0]
#     else: latitude = exif_data['GPSInfo']['GPSLatitude'][0]

#     latitude = str(latitude)
#     longitude = str(longitude)

    latitude = str(30.451468)
    longitude = str(-91.187149)
    lat_long = (latitude + "," + longitude)

    # GEOLOCATION / GIVES COUNTRY GIVEN COORDINATES
    geolocator = Nominatim(user_agent = 'Currency Converter')
    location = geolocator.reverse(lat_long)
    temp = location.raw
    country = (temp["address"]["country_code"])

    # FINDS ISO CODE FOR GIVEN COUNTRY / CURRENCY NAME
    currency_code = get_territory_currencies(country)[0]
    printing_code = ("The currency code for the location this image was taken is, " + str(currency_code))
    print(str(printing_code))

    # GETS CONVERSION RATE FROM FOUND CURRENCY TO PREFFERED CURRENCY
    c = CurrencyRates()
    cc = CurrencyConverter()
#     preference = input("Please enter the currency code you would like prices to be converted to: ")
#     conversion = c.get_rate(currency_code, preference)
    conversion=0.72

    # ELIMINATES ACCENT MARKS / FORMATS COUNTRY NAMES
    country = unidecode(country)

    # FINDS THE LANGUAGE OF THE COUNTRY FOR TRANSLATION AND USE BY OCR
    countryinf = CountryInfo(country)
    language = countryinf.languages()

    # READS TEXT IN THE IMAGE USING THE LANGUAGE GIVEN THROUGH COUNTRY INFO
    reader = easyocr.Reader(language) # this needs to run only once to load the model into memory
    result = reader.readtext(image_path, detail = 0) #####################################################3

    # TRANSLATES TEXT FOUND BY OCR TO BE PASSED TO LANGUAGE PROCESSING ALGORITHM
    translator = Translator()
    translation = translator.translate(result, dest = 'en')
    translated = []
    for translation in translation:
        translated.append(translation.text)
    
    text_read = "Sorry! No Text Detected!"
    # BANK NOTE / PRICE TAG LANGUAGE PROCESSING
    if class_label == 'P' or class_label == 'C':

        common = [1,2,5,10,20,50,100,500,1000,5000,10000,50000,100000,500000,1000000]
        denominations = []
        prediction_list = []
        most_save, index_save = 0, 0
        for k in range(len(translated)):
            most = translated.count(translated[k])
            if most > most_save: 
                most_save, index_save = most, k
        prediction_list.append(translated[index_save])

        for y in range(len(translated)):
            try:
                prediction_list.append(w2n.word_to_num(translated[y]))
            except:
                print("ValueError")

        if len(prediction_list) > 1:
            from collections import Counter
            c = Counter(prediction_list)
            prediction = c.most_common(1)
        prediction = prediction[0][0]

        for b in range(len(prediction_list)):
            if prediction_list[b] in common:
                prediction = prediction_list[b]
        print("Currency prediction :::", prediction, type(prediction))
        prediction = re.sub('\D', '', str(prediction))
        if len(prediction)>0:
            prediction_converted = float(prediction) * conversion
#             prediction_converted = cc.convert(float(prediction), 'USD', preference)
            print("Conversion occured")
            prediction_print = ("The predicted"+ str(labels) +"is: " + str(prediction) + "\nThe converted value is: " + str(prediction_converted))
            text_read = prediction_print

        else:
            print("No value detected")
#             text_read = prediction_print

        """wrong = input("Is this the bill you uploaded? If not please provide the correct denomination in the text area. ")
        if wrong != '':
            wrong = float(wrong) * conversion
            print("This is the converted currency: ", wrong)"""

    # MENU LANGUAGE PROCESSING
    if class_label == 'M':

        index = [[],[],[]]

        for i in range(len(translated)):
            if len(re.findall(r'[\d\.\d]+', translated[i])) == 1 and re.findall(r'[\d\.\d]+', translated[i]) != ['.']:
                s = re.findall(r'[\d\.\d]+', translated[i])[0]
                if s[len(s) - 1] == '.':
                    pass
                else:
                    index[0].append(translated[i-1])
                    index[1].append(re.findall(r'[\d\.\d]+', translated[i]))

        for l in range(len(index[0])):
#             index[2].append(float(index[1][l][0]) * conversion)
            index[2].append(float(index[1][l][0]) * conversion)


        print_result = ''
        for x in range(len(index[0])):
            printing_str = ("\nItem name: " + str(index[0][x]) + "\nOriginal price: " + str(index[1][x][0]) + "\nConverted price:"+ str(index[2][x]) + "\n")
            print_result += printing_str
        text_read = print_result
    return text_read
    
@app.route(f'{base_url}', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            drop_down_data = request.form.get('currency')
            global selected_conversion
            selected_conversion = drop_down_data
            print("selected drop down", drop_down_data)
            return redirect(url_for('uploaded_file',
                                    filename=filename))

    return render_template('new_index.html')


@app.route(f'{base_url}/uploads/<filename>')
def uploaded_file(filename):
    here = os.getcwd()
    image_path = os.path.join(here, app.config['UPLOAD_FOLDER'], filename)

    results = model(image_path, size=416)
    if len(results.pandas().xyxy) > 0:
        results.print()
        save_dir = os.path.join(here, app.config['UPLOAD_FOLDER'])
        results.save(save_dir=save_dir)
        def and_syntax(alist):
            if len(alist) == 1:
                alist = "".join(alist)
                return alist
            elif len(alist) == 2:
                alist = " and ".join(alist)
                return alist
            elif len(alist) > 2:
                alist[-1] = "and " + alist[-1]
                alist = ", ".join(alist)
                return alist
            else:
                return
        confidences = list(results.pandas().xyxy[0]['confidence'])
        # confidences: rounding and changing to percent, putting in function
        format_confidences = []
        for percent in confidences:
            format_confidences.append(str(round(percent*100)) + '%')
        format_confidences = and_syntax(format_confidences)

        labels = list(results.pandas().xyxy[0]['name'])
        # labels: sorting and capitalizing, putting into function
        labels = set(labels)
        labels = [emotion.capitalize() for emotion in labels]
        labels = and_syntax(labels)
        # Hudson here is the labels
        detected_text = conversion_logic(labels[0], image_path)
        print("Detected string:", detected_text)
        detected_tex = detected_text.replace('\n', '<br>')
        print("Modified:", detected_text)

        return render_template('new_result.html', confidences=format_confidences, labels=labels,
                               old_filename=filename,
                               filename=filename,conversion_str=detected_text)
    else:
        found = False
        return render_template('new_result.html', labels='No Emotion', old_filename=filename, filename=filename)


@app.route(f'{base_url}/uploads/<path:filename>')
def files(filename):
    return send_from_directory(UPLOAD_FOLDER, filename, as_attachment=True)

# define additional routes here
# for example:
# @app.route(f'{base_url}/team_members')
# def team_members():
#     return render_template('team_members.html') # would need to actually make this page

if __name__ == '__main__':
    # IMPORTANT: change url to the site where you are editing this file.
    website_url = 'cocalc19.ai-camp.dev/'
    
    print(f'Try to open\n\n    https://{website_url}' + base_url + '\n\n')
    app.run(host = '0.0.0.0', port=port, debug=True)
