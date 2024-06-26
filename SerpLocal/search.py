from playwright.sync_api import sync_playwright
from time import sleep
from .pageparser import Parser
from .datacollector import DataCollector
from .variables import DEFAULT_TIMEOUT, SLEEP_TIME, TIMEOUT_FOR_PAGE_LOAD, HEADLESS
from playwright_stealth import stealth_sync
from .variables import PROXIES
from .globals import Vars
from urllib.parse import quote

DEFAULT_TIMEOUT = DEFAULT_TIMEOUT*1000
TIMEOUT_FOR_PAGE_LOAD = TIMEOUT_FOR_PAGE_LOAD *1000


class GoogleBot:

    base_url_for_search = "https://www.google.com/search?q={query}&cs=1&filter=0"

    def initialzing_objects(self):
        """To initialze our data collector and parser objects"""

        self.parser = Parser(self.page)
        self.data_collector = DataCollector(self.parser)

    def window_scroll_down(self):
        """This function is responsible for scrolling down the window"""

        print("Scrolling down the window")
        self.page.evaluate("window.scrollTo(0, document.body.scrollHeight);")

    def check_for_loader(self) -> bool:
        """It will check whether chrome is loading the results"""

        loader_element = self.parser.get_element(
            css_selector="[aria-label='Loading...']")
        if loader_element != None and loader_element.is_visible():
            print("Chrome is loading the results")
            return True

        return False

    def scraping_data(self):
        """It will initialize scraping"""

        self.data_collector.main()

    def check_for_more_results(self):
        """This function will check for 'More results' button. And, if found, it will click on it"""

        more_results_element = self.parser.get_element(
            css_selector="a[aria-label='More results']")

        if more_results_element != None and more_results_element.is_visible():
            more_results_element.click(timeout=30000)
            print("More results button found, and clicked")

        else:
            print("More results button could not find")
            if self.check_for_loader() == True:
                print("Chrome is loading the resutls")

                pass
            else:
                print("No further results.")
                return 'break'

    def load_all_results(self):
        """It will load all the search results by scrolling down"""

        for i in range(3):
            self.window_scroll_down()

            sleep(SLEEP_TIME)

    def main(self, search):
        print("\nStartingt the bot")

        with sync_playwright() as p:

            if PROXIES != None:
                browser = p.chromium.launch(headless=HEADLESS, proxy=PROXIES)

            else:
                browser = p.chromium.launch(headless=HEADLESS)


            self.page = browser.new_page()
            stealth_sync(self.page)
            self.page.set_default_timeout(DEFAULT_TIMEOUT)

            self.initialzing_objects()

            formatted_url = self.base_url_for_search.format(query=quote(search))
            print("Opening results page")

            self.page.goto(url=formatted_url, timeout=TIMEOUT_FOR_PAGE_LOAD)

            print("Page is loaded")

            self.load_all_results()

            print("Going to scrape the data")

            self.scraping_data()

            browser.close()


def results(search):
    google_bot = GoogleBot()
    google_bot.main(search)

    return Vars.scrape
