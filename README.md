# UNICEF-Consultancy-Assessment

## 📌 Project Description

This project calculates the population-weighted coverage of two health services:

* Antenatal care (ANC4): % of women (aged 15–49) with at least 4 antenatal care visits
* Skilled birth attendance (SBA): % of deliveries attended by skilled health personnel
for countries categorized as on-track or off-track in achieving under-five mortality targets (as of 2022).
---

## ⚙️ Setup & Installation

### Requirements

- Python 3.8+
- Git

### Installation Steps

1. **Clone the repository**:
   ```bash
   git clone https://github.com/mydatadune/UNICEF-Consultancy-Assessment-Results.git
   cd UNICEF-Consultancy-Assessment-Results
2. **Configure processing**:  
Edit the `user_profile.yaml` file to configure file location and processing parameters.  
The report file name and text can be configured with the version and date configuration values using {version} and {date} embedded strings.
### How to Run the Code

In Windows execute `run_project.bat` and in Linux execute `run_project.sh`

### Description of Outputs
A report is created as a pdf file containing 3 sections:
1. **Header**:  
Configurable text that will appear at the top of the report.
2. **Visualization Chart**:  
A side-by-side bar chart comparing On-track and Off-track coverage for both analysed indicators.
3. **Footer**:  
Configurable text that will appear at the bottom of the report.

### Repository Structure

The repository includes:
* scripts directory - python code
* data directory - input files
* output directory - reports
* docs directory - documentation
* user_profile.yaml - processing configuration
* run_project scripts (.bat and .sh) - end-to-end execution of workflow
* requirements.txt - external python dependencies
* README.md - this readme file

### Positions applied for:
* Learning and Skills Data Analyst Consultant - Req. \#581598
* Administrative Data Analyst - Req. \#581696

#### Available also for positions:
* Household Survey Data Analyst Consultant - Req. \#581656
* Microdata Harmonization Consultant - Req. \#581699