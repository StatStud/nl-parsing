from selenium import webdriver
import csv
import time
import re

######################################################
################ define regex rules ##################
######################################################

def extract_text_between_br_and_i_tags(html):
    # regex pattern to match text between <br> and <i> tags
    pattern = r"<br>([^<]*?)<i>|<br>([^<]*?)</a>"

    # find all matches in the html
    matches = re.findall(pattern, html)

    # combine matches and remove blank strings
    result = [x.strip() for match in matches for x in match if x.strip()]

    return result


def extract_text_between_br_and_a_tags(html):
    # Split the HTML by <br> tag
    chunks = re.split('<br>', html)
    
    # Initialize an empty list to store the extracted text
    extracted_text = []
    
    # Loop through each chunk
    for chunk in chunks:
        # Check if an <i> tag is within the chunk
        if re.search('<i>', chunk):
            # If <i> tag is within chunk, then skip and continue
            continue
        else:
            # Otherwise, extract the text that's within the <a> tag
            match = re.search('<a[^>]*>(.*?)</a>', chunk)
            if match:
                extracted_text.append(match.group(1))
    
    return extracted_text


def get_first_entry(html):
    pattern = r'<div id="page_specific_content">(.+?)<br>'
    match = re.search(pattern, html)
    if match:
        return "<br>" + match.group(1)
    else:
        return ""
    
def extract_first_entry(html):
    result = get_first_entry(html)
    maybe = extract_text_between_br_and_a_tags(result)
    if maybe:
        return maybe
    else:
        return extract_text_between_br_and_i_tags(result)
    
######################################################
################ run web scraping ####################
######################################################

# Launch the Safari browser
driver = webdriver.Safari()

# Go to the Diseases Database website
driver.get('http://www.diseasesdatabase.com')

# Find the div container with id = "page_specific_content"
page_specific_content_div = driver.find_element('id', 'page_specific_content')

# Find the table with id="alphaidx" within the div container
alphaidx_table = page_specific_content_div.find_element('id', 'alphaidx')

i = 0
# Loop through all the a href links inside the table
with open('output.csv', mode='a', newline='') as csv_file:

    # Create a CSV writer object
    writer = csv.writer(csv_file)
    # Write the header row if the file is empty
    if csv_file.tell() == 0:
        writer.writerow(['node_name'])

    for link in alphaidx_table.find_elements('tag name', 'a'):
        # Follow the link
        link.click()
        # Sleep for 3 seconds
        time.sleep(3) 

        
        # Find the div id='page_specific_content'
        link_page_specific_content_div = driver.find_element('id', 'page_specific_content')
        raw_html = link_page_specific_content_div.get_attribute('outerHTML')

        lst1 = extract_text_between_br_and_i_tags(raw_html)
        lst2 = extract_text_between_br_and_a_tags(raw_html)
        lst3 = extract_first_entry(raw_html)

        combined_list = lst1 + lst2 + lst3

        # Write each element in the combined_list as a new row
        for element in combined_list:
            writer.writerow([element])

        # Go back to the previous page
        i += 1
        print(f"finished list {i} out of 26")
        driver.back()

# Close the browser
driver.quit()
