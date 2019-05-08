import base64
import json
import io
import os
import logging
import datetime
import re

import sqlite3
from sqlite3 import Error

import pandas as pd
import requests
from rq import get_current_job
import tqdm
from threesixty import ThreeSixtyGiving

from .cache import get_cache, get_from_cache, save_to_cache, get_db_connection
from .utils import get_fileid, charity_number_to_org_id
from .registry import fetch_reg_file, get_reg_file_from_url

FTC_URL = 'https://findthatcharity.uk/orgid/{}.json'
CH_URL = 'http://data.companieshouse.gov.uk/doc/company/{}.json'
PC_URL = 'https://postcodes.findthatcharity.uk/postcodes/{}.json'

# config
# schemes with data on findthatcharity
BELGIAN_SCHEMES = ['BE-BCE_KBO']
FTC_SCHEMES = ["GB-CHC", "GB-NIC", "GB-SC", "GB-COH"]
KNOWN_SCHEMES = FTC_SCHEMES + BELGIAN_SCHEMES
UK_POSTCODE_FIELDS = ['ctry', 'cty', 'laua', 'pcon', 'rgn', 'imd', 'ru11ind',
                   'oac11', 'lat', 'long']  # fields to care about from the postcodes)


def get_dataframe_from_file(filename, contents, date=None, expire_days=(2 * (365/12))):
    fileid = get_fileid(contents, filename, date)

    # 2. Check cache for file
    df = get_from_cache(fileid)
    if df is not None:
        return (fileid, filename)

    # 3. Fetch and prepare the data
    df = None
    cache = prepare_lookup_cache()
    job = get_current_job()

    data_preparation = DataPreparation(
        df, cache, job, filename=filename, contents=contents)
    data_preparation.stages = [LoadDatasetFromFile] + data_preparation.stages
    df = data_preparation.run()

    # 4. set expiry time
    metadata = {
        "expires": (datetime.datetime.now() + datetime.timedelta(expire_days)).isoformat()
    }

    # 5. save to cache
    save_to_cache(fileid, df, metadata=metadata)  # dataframe
    return (fileid, filename)

def get_dataframe_from_url(url):

    # 1. Work out the file id
    headers = fetch_reg_file(url, 'HEAD')

    # 2. Get the registry entry for the file (if available)
    registry = get_reg_file_from_url(url)
    if registry and registry.get("identifier"):
        fileid = registry.get("identifier")
    else:
        # work out the version of the file
        # https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Last-Modified
        last_modified = headers.get("ETag", headers.get("Last-Modified"))

        fileid = get_fileid(None, url, last_modified)
    
    # 2. Check cache for file
    df = get_from_cache(fileid)
    if df is not None:
        print("using cache")
        return (fileid, url, headers)

    # 3. Fetch and prepare the data
    df = None
    cache = prepare_lookup_cache()
    job = get_current_job()

    data_preparation = DataPreparation(df, cache, job, url=url)
    data_preparation.stages = [LoadDatasetFromURL] + data_preparation.stages
    df = data_preparation.run()

    # 4. Get metadata about the file
    metadata = {
        "headers": headers,
        "url": url,
    }
    if registry:
        metadata["registry_entry"] = registry

    # 5. save to cache
    save_to_cache(fileid, df, metadata=metadata)  # dataframe

    return (fileid, url, headers)


def prepare_lookup_cache(cache=None):
    if cache is None:
        cache = get_cache()

    if not cache.exists("geocodes"):
        for g, name in fetch_geocodes().items():
            cache.hset("geocodes", g, name)
    return cache


def fetch_geocodes():
    r = requests.get("https://postcodes.findthatcharity.uk/areas/names.csv",
                     params={"types": ",".join(UK_POSTCODE_FIELDS)})
    pc_names = pd.read_csv(io.StringIO(r.text)).set_index(
        ["type", "code"]).sort_index()
    geocodes = pc_names["name"].to_dict()
    geocodes = {"-".join(c): d for c, d in geocodes.items()}
    return geocodes


class DataPreparation(object):
    
    def __init__(self, df, cache=None, job=None, **kwargs):
        self.stages = [
            CheckColumnNames,
            CheckColumnsExist,
            CheckColumnTypes,
            AddExtraColumns,
            CleanRecipientIdentifiers,
            LookupCharityDetails,
            LookupCompanyDetails,
            LookupBelgianDetails,
            MergeCompanyAndCharityDetails,
            FetchPostcodes,
            MergeGeoData,
            AddExtraFieldsExternal,
        ]
        self.df = df
        self.cache = cache
        self.job = job
        self.attributes = kwargs

    def _progress_job(self, stage_id, progress=None):
        if not self.job:
            return
        self.job.meta['progress']["stage"] = stage_id
        self.job.meta['progress']["progress"] = progress
        self.job.save_meta()

    def _setup_job_meta(self):
        if not self.job:
            return
        self.job.meta['stages'] = [s.name for s in self.stages]
        self.job.meta['progress'] = {"stage": 0, "progress": None}
        self.job.save_meta()

    def run(self):
        df = None
        self._setup_job_meta()
        for k, Stage in enumerate(self.stages):
            stage = Stage(df, self.cache, self.job, **self.attributes)
            logging.info(stage.name)
            df = stage.run()
            self._progress_job(k)
        return df


class DataPreparationStage(object):

    def __init__(self, df, cache, job, **kwargs):
        self.df = df
        self.cache = cache
        self.job = job
        self.attributes = kwargs

    def _progress_job(self, item_key, total_items):
        if not self.job:
            return
        self.job.meta['progress']["progress"] = (item_key, total_items)
        self.job.save_meta()

    def run(self):
        # subclasses should implement a `run()` method
        # which returns an altered version of the dataframe
        return self.df


class LoadDatasetFromURL(DataPreparationStage):

    name = 'Load data to be prepared from an URL'

    def run(self):

        if not self.attributes.get("url"):
            return self.df

        url = self.attributes.get("url")
        self.df = ThreeSixtyGiving.from_url(url).to_pandas()

        return self.df

class LoadDatasetFromFile(DataPreparationStage):

    name = 'Load data to be prepared from a file'

    def run(self):

        if not self.attributes.get("contents") or not self.attributes.get("filename"):
            return self.df

        contents = self.attributes.get("contents")
        filename = self.attributes.get("filename")

        if isinstance(contents, str):
            # if it's a string we assume it's dataurl/base64 encoded
            content_type, content_string = contents.split(',')
            contents = base64.b64decode(content_string)

        if filename.endswith("csv"):
            # Assume that the user uploaded a CSV file
            self.df = ThreeSixtyGiving.from_csv(io.BytesIO(contents)).to_pandas()
        elif filename.endswith("xls") or filename.endswith("xlsx"):
            # Assume that the user uploaded an excel file
            self.df = ThreeSixtyGiving.from_excel(
                io.BytesIO(contents)).to_pandas()
        elif filename.endswith("json"):
            # Assume that the user uploaded a json file
            self.df = ThreeSixtyGiving.from_json(
                io.BytesIO(contents)).to_pandas()

        return self.df


class CheckColumnNames(DataPreparationStage):
    # check column names for typos

    name = 'Check column names'
    columns_to_check = [
        'Amount Awarded', 'Funding Org:0:Name', 'Award Date',
        'Recipient Org:0:Name', 'Recipient Org:0:Identifier'
    ]

    def run(self):
        renames = {}
        for c in self.df.columns:
            for w in self.columns_to_check:
                if c.replace(" ", "").lower() == w.replace(" ", "").lower() and c != w:
                    renames[c] = w
                # @TODO: could include a replacement of (eg) "Recipient Org:Name" with "Recipient Org:0:Name"
        self.df = self.df.rename(columns=renames)
        return self.df

class CheckColumnsExist(CheckColumnNames):

    name = 'Check columns exist'

    def run(self):
        for c in self.columns_to_check:
            if c not in self.df.columns:
                raise ValueError("Column {} not found in data. Columns: [{}]".format(
                    c, ", ".join(self.df.columns)
                ))
        return self.df

class CheckColumnTypes(DataPreparationStage):

    name = 'Check column types'
    columns_to_check = {
        "Amount Awarded": lambda x: x.astype(float),
        "Funding Org:0:Identifier": lambda x: x.str.strip(),
        "Funding Org:0:Name": lambda x: x.str.strip(),
        "Recipient Org:0:Name": lambda x: x.str.strip(),
        "Recipient Org:0:Identifier": lambda x: x.str.strip(),
        "Award Date": lambda x: pd.to_datetime(x),
    }

    def run(self):
        for c, func in self.columns_to_check.items():
            if c in self.df.columns:
                self.df.loc[:, c] = func(self.df[c])
        return self.df

class AddExtraColumns(DataPreparationStage):

    name = 'Add extra columns'

    def run(self):
        self.df.loc[:, "Award Date:Year"] = self.df["Award Date"].dt.year
        self.df.loc[:, "Recipient Org:0:Identifier:Scheme"] = self.df["Recipient Org:0:Identifier"].apply(
            lambda x: "360G" if x.startswith("360G-") else "-".join(x.split("-")[:2])
        )
        return self.df

class CleanRecipientIdentifiers(DataPreparationStage):

    name = 'Clean recipient identifiers'

    def run(self):
        # default is use existing identifier
        self.df.loc[
            self.df["Recipient Org:0:Identifier:Scheme"].isin(KNOWN_SCHEMES),
            "Recipient Org:0:Identifier:Clean"
        ] = self.df.loc[
            self.df["Recipient Org:0:Identifier:Scheme"].isin(KNOWN_SCHEMES),
            "Recipient Org:0:Identifier"
        ]

        # add company number for those with it
        if "Recipient Org:0:Company Number" in self.df.columns:
            self.df.loc[
                self.df["Recipient Org:0:Company Number"].notnull(),
                "Recipient Org:0:Identifier:Clean"
            ] = self.df.loc[:, "Recipient Org:0:Identifier:Clean"].fillna(
                self.df["Recipient Org:0:Company Number"].apply("GB-COH-{}".format)
            )

        # add charity number for those with it
        if "Recipient Org:0:Charity Number" in self.df.columns:
            self.df.loc[:, "Recipient Org:0:Identifier:Clean"] = self.df.loc[:, "Recipient Org:0:Identifier:Clean"].fillna(
                self.df["Recipient Org:0:Charity Number"].apply(charity_number_to_org_id)
            )
        
        # overwrite the identifier scheme using the new identifiers
        # @TODO: this doesn't work well at the moment - seems to lose lots of identifiers
        self.df.loc[:, "Recipient Org:0:Identifier:Scheme"] = self.df["Recipient Org:0:Identifier:Clean"].apply(
            lambda x: ("360G" if x.startswith(
                "360G-") else "-".join(x.split("-")[:2])) if isinstance(x, str) else None
        ).fillna(self.df["Recipient Org:0:Identifier:Scheme"])

        return self.df

class LookupCharityDetails(DataPreparationStage):

    name = 'Look up charity data'
    ftc_url=FTC_URL

    # utils
    def _get_charity(self, orgid):
        if self.cache.hexists("charity", orgid):
            return json.loads(self.cache.hget("charity", orgid))
        return requests.get(self.ftc_url.format(orgid)).json()

    def run(self):

        orgids = self.df.loc[
            self.df["Recipient Org:0:Identifier:Scheme"].isin(FTC_SCHEMES),
            "Recipient Org:0:Identifier:Clean"
        ].dropna().unique()
        print("Finding details for {} charities".format(len(orgids)))
        for k, orgid in tqdm.tqdm(enumerate(orgids)):
            self._progress_job(k+1, len(orgids))
            try:
                self.cache.hset("charity", orgid, json.dumps(self._get_charity(orgid)))
            except ValueError:
                pass

        return self.df

class LookupCompanyDetails(DataPreparationStage):

    name = 'Look up company data'
    ch_url = CH_URL

    def _get_company(self, orgid):
        if self.cache.hexists("company", orgid):
            return json.loads(self.cache.hget("company", orgid))
        return requests.get(self.ch_url.format(orgid.replace("GB-COH-", ""))).json()

    def _get_orgid_index(self):
        # find records where the ID has already been found in charity lookup
        return list(self.cache.hkeys("charity"))

    def run(self):
        company_orgids = self.df.loc[
            ~self.df["Recipient Org:0:Identifier:Clean"].isin(self._get_orgid_index()) &
            (self.df["Recipient Org:0:Identifier:Scheme"] == "GB-COH"),
            "Recipient Org:0:Identifier:Clean"
        ].unique()
        print("Finding details for {} companies".format(len(company_orgids)))
        for k, orgid in tqdm.tqdm(enumerate(company_orgids)):
            self._progress_job(k+1, len(company_orgids))
            try:
                self.cache.hset("company", orgid, json.dumps(self._get_company(orgid)))
            except ValueError:
                pass

        return self.df

class LookupBelgianDetails(DataPreparationStage):

    name = 'Look up Belgian company data'
    cache_name = 'be_company'
    base_url = 'https://data.be/fr/company/official/BE_{}/social'
    second_url = 'https://www.socialsecurity.be/app014/wrep/rep/gp/jsp/fr/REPGPdata.jsp'
    regex = re.compile('<td .+>Code importance:</td>[\n\t]*<td .+>([^<]+)</td>')
    regex_name = re.compile('<td .+>D&eacute;nomination:</td>[\n\t]*<td .+>([^<]+)</td>')

    def _find_nbr_employees(self, enterprise_number):
        conn_sql3 = get_db_connection()
        sql3 = conn_sql3.cursor() 
        result = sql3.execute('SELECT Employees FROM employer_status WHERE EnterpriseNumber=?', (enterprise_number,)).fetchone()

        if result is not None:
            conn_sql3.close()
            return result[0]
        else:
            ''' fetching is not yet working, should have full db'''
            headers = {
                'User-agent': 'Mozilla/5.0 (X11; U; Linux i686; en-US; rv:1.9.0.1) Gecko/2008071615 Fedora/3.0.1-1.fc9 Firefox/3.0.1', 
                'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Cache-Control': 'no-cache'
            }
            s = requests.Session()
            r = s.get(self.base_url.format(enterprise_number.replace('.', '')), headers=headers)
            r2 = s.get(self.second_url, headers=headers, cookies=r.cookies)
            print(r2.request.headers)

            if r2.text.find('(201100) - L\'employeur demand&eacute; n\'existe ni dans le R&eacute;pertoire ONSS, ni dans celui de l\'ORPSS.') > -1:
                print('{} not found in ONSS db'.format(enterprise_number))
                nbr_employees = '0'
            else:
                name = self.regex_name.findall(r2.text)[0]
                nbr_employees = self.regex.findall(r2.text)[0]
                print('{} found with name {}'.format(enterprise_number, name))

            #sql3.execute('INSERT INTO employer_status(EnterpriseNumber, Employees) VALUES(?, ?)', (enterprise_number,nbr_employees))
            conn_sql3.commit()
            conn_sql3.close()
            return nbr_employees
    
    def _get_kbo_data(self, enterprise_number):
        try: 
            conn_sql3 = get_db_connection()
            sql3 = conn_sql3.cursor() 
            result = sql3.execute('SELECT * FROM enterprise WHERE EnterpriseNumber=?', (enterprise_number,)).fetchone()
            conn_sql3.close()
            
            '''
            db_fields = ['EntityNumber', 'Status',  'JuridicalSituation', 'TypeOfEnterprise', 'JuridicalForm', 
                'StartDate', 'TypeOfAddress', 'Zipcode', 'MunicipalityNL', 'MunicipalityFR', 'StreetNL', 'StreetFR', 'HouseNumber', 
                'Box', 'municipality_nis_code', 'Denomination', 'OtherDenomination', 'assets', 'operatingIncome', 
                'operatingCharges', 'averageFTE', 'averageFTEMen', 'averageFTEWomen']'''
            db_fields = ['EnterpriseNumber', 'Status',  'JuridicalSituation', 'TypeOfEnterprise', 'JuridicalForm', 
                'StartDate']
            
            result = dict(zip(db_fields, list(result)))
        except TypeError:
            print('Could not find company {}'.format(enterprise_number))
        return result
    
    def _get_company(self, orgid):
        if self.cache.hexists(self.cache_name, orgid):
            return json.loads(self.cache.hget(self.cache_name, orgid))
        else:
            db_orgid = orgid.replace("BE-BCE_KBO-", "")
            db_orgid = db_orgid[:4]+'.'+db_orgid[4:7]+'.'+db_orgid[7:]
            
            result = self._get_kbo_data(db_orgid)
            if result is not None:
                result['nbr_employees'] = self._find_nbr_employees(db_orgid)
            return result

    def run(self):
        company_orgids = self.df.loc[
            self.df["Recipient Org:0:Identifier:Scheme"].isin(BELGIAN_SCHEMES),
            "Recipient Org:0:Identifier:Clean"
        ].unique()
        print("Finding details for {} belgian companies".format(len(company_orgids)))
        for k, orgid in tqdm.tqdm(enumerate(company_orgids)):
            self._progress_job(k+1, len(company_orgids))
            try:
                self.cache.hset(self.cache_name, orgid, json.dumps(self._get_company(orgid)))
            except ValueError:
                pass

        return self.df


class MergeCompanyAndCharityDetails(DataPreparationStage):

    name = 'Add charity and company details to data'

    COMPANY_REPLACE = {
        "PRI/LBG/NSC (Private, Limited by guarantee, no share capital, use of 'Limited' exemption)": "Company Limited by Guarantee",
        "PRI/LTD BY GUAR/NSC (Private, limited by guarantee, no share capital)": "Company Limited by Guarantee",
        "PRIV LTD SECT. 30 (Private limited company, section 30 of the Companies Act)": "Private Limited Company",
    }  # replacement values for companycategory

    def _get_org_type(self, id):
        if id.startswith("S") or id.startswith("GB-SC-"):
            return "Registered Charity (Scotland)"
        elif id.startswith("N") or id.startswith("GB-NIC-"):
            return "Registered Charity (NI)"
        return "Registered Charity (E&W)"
    
    def _belgian_org_type(self, code):
        org_type = {'017': 'ASBL/VZW', '028': 'ISBL/IZW', '117': 'ASBL de droit public / VZW van publiek recht', 
            '125': 'AISBL/IVZW', '325': 'AISBL de droit public / IVZW van publiek recht', 
            '026': 'Fondation privée / Private stichting', '029': 'Fondation d\'utilité publique / Stichting van openbaar nut',
            '411': 'Ville / commune / Stad / gemeente', '412': 'Centre public d\'action sociale / Openbaar centrum voor maatschappelijk welzijn'
        }
        if code in org_type:
            return 'Belgian ' + org_type[code]
        else:
            print('not found: {}'.format(code))
            return 'Other form of organisation'

    def _create_orgid_df(self):

        orgids = self.df["Recipient Org:0:Identifier:Clean"].unique()
        charity_rows = []
        for k, c in self.cache.hscan_iter("charity"):
            c = json.loads(c)
            if k.decode("utf8") in orgids:
                charity_rows.append({
                    "orgid": k.decode("utf8"),
                    "charity_number": c.get('id'),
                    "company_number": c.get("company_number")[0].get("number") if c.get("company_number") else None,
                    "date_registered": c.get("date_registered"),
                    "date_removed": c.get("date_removed"),
                    "postcode": c.get("geo", {}).get("postcode"),
                    "latest_income": c.get("latest_income"),
                    "org_type": self._get_org_type(c.get("id")),
                })

        if not charity_rows:
            return None

        orgid_df = pd.DataFrame(charity_rows).set_index("orgid")

        orgid_df.loc[:, "date_registered"] = pd.to_datetime(
            orgid_df.loc[:, "date_registered"])
        orgid_df.loc[:, "date_removed"] = pd.to_datetime(
            orgid_df.loc[:, "date_removed"])

        return orgid_df

    def _create_company_df(self):

        orgids = self.df["Recipient Org:0:Identifier:Clean"].unique()

        company_rows = []
        for k, c in self.cache.hscan_iter("company"):
            c = json.loads(c)
            if k.decode("utf8") in orgids:
                company = c.get("primaryTopic", {})
                company = {} if company is None else company
                address = c.get("primaryTopic", {}).get("RegAddress", {})
                address = {} if address is None else address
                company_rows.append({
                    "orgid": k.decode("utf8"),
                    "charity_number": None,
                    "company_number": company.get("CompanyNumber"),
                    "date_registered": company.get("IncorporationDate"),
                    "date_removed": company.get("DissolutionDate"),
                    "postcode": address.get("Postcode"),
                    "latest_income": None,
                    "org_type": self.COMPANY_REPLACE.get(company.get("CompanyCategory"), company.get("CompanyCategory")),
                })
        
        if not company_rows:
            return None

        companies_df = pd.DataFrame(company_rows).set_index("orgid")
        companies_df.loc[:, "date_registered"] = pd.to_datetime(
            companies_df.loc[:, "date_registered"], dayfirst=True)
        companies_df.loc[:, "date_removed"] = pd.to_datetime(
            companies_df.loc[:, "date_removed"], dayfirst=True)

        return companies_df
    
    def _create_kbo_bce_df(self):

        orgids = self.df["Recipient Org:0:Identifier:Clean"].unique()
        rows = []
        for k, c in self.cache.hscan_iter("be_company"):
            c = json.loads(c)
            if c is None:
                print('No info on {}'.format(k.decode("utf8")))
            else:
                if k.decode("utf8") in orgids:
                    rows.append({
                        "orgid": k.decode("utf8"),
                        "charity_number": None,
                        "company_number": None,
                        "date_registered": c.get("StartDate"),
                        "date_removed": None,
                        "postcode": None, #c.get("Zipcode"),
                        "latest_income": None, #c.get("operatingIncome"),
                        "org_type": self._belgian_org_type(c.get("JuridicalForm")),
                    })

        if not rows:
            return None

        orgid_df = pd.DataFrame(rows).set_index("orgid")

        orgid_df.loc[:, "date_registered"] = pd.to_datetime(
            orgid_df.loc[:, "date_registered"])
        orgid_df.loc[:, "date_removed"] = pd.to_datetime(
            orgid_df.loc[:, "date_removed"])

        return orgid_df

    def run(self):

        # create orgid dataframes
        orgid_df = self._create_orgid_df()
        companies_df = self._create_company_df()
        kbo_bce_df = self._create_kbo_bce_df()

        if isinstance(orgid_df, pd.DataFrame) and isinstance(companies_df, pd.DataFrame):
            orgid_df = pd.concat([orgid_df, companies_df], sort=False)
        elif isinstance(companies_df, pd.DataFrame):
            orgid_df = companies_df
        
        if isinstance(orgid_df, pd.DataFrame) and isinstance(kbo_bce_df, pd.DataFrame):
            orgid_df = pd.concat([orgid_df, kbo_bce_df], sort=False)
        elif isinstance(kbo_bce_df, pd.DataFrame):
            orgid_df = kbo_bce_df
        
        if not isinstance(orgid_df, pd.DataFrame):
            return self.df

        # drop any duplicates
        orgid_df = orgid_df[~orgid_df.index.duplicated(keep='first')]

        # create some extra fields
        orgid_df.loc[:, "age"] = pd.datetime.now() - orgid_df["date_registered"]
        orgid_df.loc[:, "latest_income"] = orgid_df["latest_income"].astype(float)

        # merge org details into main dataframe
        self.df = self.df.join(orgid_df.rename(columns=lambda x: "__org_" + x),
                     on="Recipient Org:0:Identifier:Clean", how="left")
        
        # Add type 'individual'
        if 'recipientOrganization.0.Type' in self.df.columns:
            self.df.loc[self.df['recipientOrganization.0.Type'] == 'IND', '__org_org_type'] = 'Individuals'

        return self.df

class FetchPostcodes(DataPreparationStage):

    name = 'Look up postcode data'
    pc_url = PC_URL

    def _normalise_country_name(self):
        conn_sql3 = get_db_connection()
        sql3 = conn_sql3.cursor()
        country_dict = {}
        for row in sql3.execute('SELECT CountryName, ISO2, ISO3 FROM country_code'):
            country_dict[row[0].lower()] = row[1].lower()
            country_dict[row[2].lower()] = row[1].lower()
        conn_sql3.close()

        # Add rules for uk
        country_dict['uk'] = 'gb'
        country_dict['england'] = 'gb'
        country_dict['wales'] = 'gb'
        country_dict['scotland'] = 'gb'
        country_dict['northern ireland'] = 'gb'

        if 'Recipient Org:0:Country' in self.df.columns:
            self.df['Recipient Org:0:Country'] = self.df['Recipient Org:0:Country'].apply(lambda x : country_dict[x.lower()] if x.lower() in country_dict else x.lower())
        else:
            self.df['Recipient Org:0:Country'] = 'gb'

    def _get_postcode(self, pc):
        hkey = pc['Recipient Org:0:hkey pc']
        if self.cache.hexists("postcode", hkey):
            return json.loads(self.cache.hget("postcode", hkey))
        # @TODO: postcode cleaning and formatting
        else:
            if pc['Recipient Org:0:Country'] == 'gb':
                return requests.get(self.pc_url.format(pc['Recipient Org:0:Postal Code'])).json()
            elif pc['Recipient Org:0:Country'] == 'be':

                try: 
                    conn_sql3 = get_db_connection()
                    sql3 = conn_sql3.cursor() 
                    result = sql3.execute('SELECT postcode, city, long, lat, province FROM postcode_geo WHERE postcode=?', (pc['Recipient Org:0:Postal Code'],)).fetchone()
                    conn_sql3.close()
                    db_fields = ['postcode',  'city', 'long', 'lat', 'province']
                    
                    return dict(zip(db_fields, list(result)))
                except:
                    print('postcode not found', pc['Recipient Org:0:Postal Code'])
                    return None

    def run(self):
        # check for recipient org postcode field first
        if "Recipient Org:0:Postal Code" in self.df.columns and "__org_postcode" in self.df.columns:
            self.df.loc[:, "Recipient Org:0:Postal Code"] = self.df.loc[:, "Recipient Org:0:Postal Code"].fillna(self.df["__org_postcode"])
        elif "__org_postcode" in self.df.columns:
            self.df.loc[:, "Recipient Org:0:Postal Code"] = self.df["__org_postcode"]
        elif "Recipient Org:0:Postal Code" not in self.df.columns:
            self.df.loc[:, "Recipient Org:0:Postal Code"] = None

        # fetch postcode data
        self._normalise_country_name()
        self.df['Recipient Org:0:hkey pc'] = self.df['Recipient Org:0:Country'] + ' - ' + self.df["Recipient Org:0:Postal Code"]
        postcodes = self.df.loc[:, ['Recipient Org:0:hkey pc', 'Recipient Org:0:Country', "Recipient Org:0:Postal Code"]].dropna().drop_duplicates()
        print("Finding details for {} postcodes".format(len(postcodes)))
        for k, pc in tqdm.tqdm(postcodes.iterrows()):
            self._progress_job(k+1, len(postcodes))
            try:
                self.cache.hset("postcode", pc['Recipient Org:0:hkey pc'], json.dumps(self._get_postcode(pc)))
            except json.JSONDecodeError:
                continue

        return self.df

class MergeGeoData(DataPreparationStage):

    name = 'Add geo data'
    UK_POSTCODE_FIELDS = UK_POSTCODE_FIELDS

    def _convert_geocode(self, areatype, geocode_code):
        geocode_name = self.cache.hget(
            "geocodes", "-".join([areatype, str(geocode_code)]))
        if not geocode_name:
            return geocode_code
        if isinstance(geocode_name, bytes):
            return geocode_name.decode("utf8")
        return geocode_name

    def _belgian_pc(self, k, c):
        return {
            'Recipient Org:0:hkey pc': k.decode("utf8"),
            'ctry': 'Belgium', 
            'rgn': c.get('province'), 
            'long': c.get('long'), 
            'lat': c.get('lat')
        }

    def _uk_pc(self, k, c):
        return {
            'Recipient Org:0:hkey pc': k.decode("utf8"),
            'ctry': self.cache.hget('geocodes', 'ctry-'+c.get("data", {}).get("attributes", {}).get('ctry')).decode("utf8"), 
            'rgn': self.cache.hget('geocodes', 'rgn-'+c.get("data", {}).get("attributes", {}).get('rgn')).decode("utf8").replace('(pseudo) ', ''), 
            'long': c.get("data", {}).get("attributes", {}).get('long'), 
            'lat': c.get("data", {}).get("attributes", {}).get('lat')
        }

    def _create_postcode_df(self):
        postcodes = self.df["Recipient Org:0:hkey pc"].unique()
        postcode_rows = []
        for k, c in self.cache.hscan_iter("postcode"):
            c = json.loads(c)
            if k.decode("utf8") in postcodes:
                try:
                    if k.decode("utf8").startswith('be'):
                        postcode_rows.append(self._belgian_pc(k, c))
                    elif k.decode("utf8").startswith('gb'):
                        postcode_rows.append(self._uk_pc(k, c))
                except:
                    print('Broken record for ', k.decode("utf8"))
        
        if not postcode_rows:
            return None

        postcode_df = pd.DataFrame(postcode_rows).set_index('Recipient Org:0:hkey pc')
        return postcode_df

    def run(self):
        
        postcode_df = self._create_postcode_df()
        if postcode_df is not None:
            self.df = self.df.join(postcode_df.rename(columns=lambda x: "__geo_" + x),
                        on="Recipient Org:0:hkey pc", how="left")
        return self.df

class AddExtraFieldsExternal(DataPreparationStage):

    name = 'Add extra fields from external data'

    # Bins used for numeric fields
    AMOUNT_BINS = [-1, 500, 1000, 2000, 5000, 10000, 100000, 1000000, float("inf")]
    AMOUNT_BIN_LABELS = ["Under €500", "€500 - €1k", "€1k - €2k", "€2k - €5k", "€5k - €10k",
                        "€10k - €100k", "€100k - €1m", "Over €1m"]
    INCOME_BINS = [-1, 10000, 100000, 1000000, 10000000, float("inf")]
    INCOME_BIN_LABELS = ["Under €10k", "€10k - €100k",
                        "€100k - €1m", "€1m - €10m", "Over €10m"]
    AGE_BINS = pd.to_timedelta(
        [x * 365 for x in [-1, 1, 2, 5, 10, 25, 200]], unit="D")
    AGE_BIN_LABELS = ["Under 1 year", "1-2 years", "2-5 years",
                    "5-10 years", "10-25 years", "Over 25 years"]

    def run(self):
        self.df.loc[:, "Amount Awarded:Bands"] = pd.cut(
            self.df["Amount Awarded"], bins=self.AMOUNT_BINS, labels=self.AMOUNT_BIN_LABELS)

        if "__org_latest_income" in self.df.columns:
            self.df.loc[:, "__org_latest_income_bands"] = pd.cut(self.df["__org_latest_income"].astype(float),
                                                                 bins=self.INCOME_BINS, labels=self.INCOME_BIN_LABELS)

        if "__org_age" in self.df.columns:
            self.df.loc[:, "__org_age_bands"] = pd.cut(
                self.df["__org_age"], bins=self.AGE_BINS, labels=self.AGE_BIN_LABELS)

        if "Grant Programme:0:Title" not in self.df.columns:
            self.df.loc[:, "Grant Programme:0:Title"] = "All grants"

        return self.df
