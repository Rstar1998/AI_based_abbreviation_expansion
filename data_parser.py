import xml.etree.ElementTree as ET
import pandas as pd

# Parse the XML content
tree = ET.parse("desc2024.xml")  # Replace with the path to your XML file
root = tree.getroot()

# Initialize a list to hold rows of data
records = []

# Iterate over Descriptor Records
for descriptor in root.findall("DescriptorRecord"):
    descriptor_id = descriptor.find("DescriptorUI").text
    descriptor_name = descriptor.find("DescriptorName/String").text
    
    # Extract Creation, Revision, and Establishment Dates
    creation_date = "-".join([descriptor.find(f"DateCreated/{tag}").text for tag in ["Year", "Month", "Day"]])
    revision_date = "-".join([descriptor.find(f"DateRevised/{tag}").text for tag in ["Year", "Month", "Day"]])
    establishment_date = "-".join([descriptor.find(f"DateEstablished/{tag}").text for tag in ["Year", "Month", "Day"]])

    # Iterate over Allowable Qualifiers
    qualifiers = descriptor.find("AllowableQualifiersList")
    if qualifiers is not None:
        for qualifier in qualifiers.findall("AllowableQualifier"):
            qualifier_id = qualifier.find("QualifierReferredTo/QualifierUI").text
            qualifier_name = qualifier.find("QualifierReferredTo/QualifierName/String").text
            qualifier_abbreviation = qualifier.find("Abbreviation").text

            # Add a row for each qualifier
            records.append({
                "Descriptor ID": descriptor_id,
                "Name": descriptor_name,
                "Creation Date": creation_date,
                "Revision Date": revision_date,
                "Establishment Date": establishment_date,
                "Qualifier ID": qualifier_id,
                "Qualifier Name": qualifier_name,
                "Qualifier Abbreviation": qualifier_abbreviation,
            })
    else:
        # Add a row even if no qualifiers exist
        records.append({
            "Descriptor ID": descriptor_id,
            "Name": descriptor_name,
            "Creation Date": creation_date,
            "Revision Date": revision_date,
            "Establishment Date": establishment_date,
            "Qualifier ID": None,
            "Qualifier Name": None,
            "Qualifier Abbreviation": None,
        })

# Convert the records into a DataFrame
df = pd.DataFrame(records)

print(df.head())  # Display the first few rows of the DataFrame
