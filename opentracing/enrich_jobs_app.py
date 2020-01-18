import sys
import logging
import requests
from jaeger_client import Config
from transitions import Machine

logging.getLogger('requests').setLevel(logging.CRITICAL)
logging.getLogger('urllib3').setLevel(logging.CRITICAL)
logging.getLogger('transitions').setLevel(logging.CRITICAL)


class DistributedSystem(object):

    # The states
    states = ['request', 'description', 'location', 'published']
    init_state = states[0]

    transitions = [
        {'trigger': 'get_job', 'source': 'request', 'dest': 'description',
         'before': 'action_handle_request', 'after': 'action_get_job'},
        {'trigger': 'get_location', 'source': 'description', 'dest': 'location', 'before': 'action_get_location'},
        {'trigger': 'publish', 'source': 'location', 'dest': 'published', 'before': 'action_publish'},
    ]

    def __init__(self):
        self.span_stack = []
        self.tracer = None
        self.queue = []

    def _get_tracer(self):
        return self.tracer

    def action_handle_request(self):
        op_name = sys._getframe().f_code.co_name
        # In reality, each service should init its tracer
        self.tracer = init_tracer('github-job-analysis')
        with self.tracer.start_span(operation_name=op_name) as span:
            span.set_tag('my-tag', '60')
            self.span_stack.append(span)

    def action_get_job(self):
        """Adapted from https://opentracing.io/guides/python/quickstart/"""
        # Initialize the tracer and send the service name to trace
        op_name = sys._getframe().f_code.co_name
        curr_span = self.span_stack[-1]

        homepages = []
        with self._get_tracer().start_span(op_name, child_of=curr_span) as span:
            res = requests.get('https://jobs.github.com/positions.json?description=python')
            span.set_tag('jobs-count', len(res.json()))
            results = res.json()[:5]
            for result in results:
                with self._get_tracer().start_span(result['company'], child_of=span) as site_span:
                    print(f'Getting website for {result["company"]}: {result["company_url"]}')
                    try:
                        status = requests.get(result['company_url'])
                        if status.status_code == 200:
                            homepages.append((result['company'], result['location']))
                            site_span.set_tag('status', 'Succeed')
                        else:
                            print(f'Unable to get site for {result["company"]}')
                            site_span.set_tag('status', 'Failed')
                    except:
                        print('Unable to get site for %s' % result['company'])
        # simulate sending the results to a Message Queuing system
        self.queue.append((op_name, homepages))

    def action_get_location(self):
        op_name = sys._getframe().f_code.co_name
        curr_span = self.span_stack[-1]
        _, input = self.queue.pop()

        print('Input to process: ', input)
        if not input:
            return

        with self._get_tracer().start_span(operation_name=op_name, child_of=curr_span) as span:
            span.set_tag('locations_received', len(input))

            company, city = input[0]
            json = requests.\
                get(f'https://en.wikipedia.org/w/api.php?action=opensearch&search={city}&format=json').json()

            if not input or len(json) < 4:
                return

            for link in json[3]:
                with self._get_tracer().start_span(link, child_of=span) as site_span:
                    print(f'Getting link: {link}')
                    try:
                        status = requests.get(link)
                        if status.status_code == 200:
                            site_span.set_tag('status', 'Succeed')
                        else:
                            print(f'Unable to get link {link}')
                            site_span.set_tag('status', 'Failed')
                    except:
                        print('Unable to get site for %s' % link)

    def action_publish(self):
        op_name = sys._getframe().f_code.co_name
        curr_span = self.span_stack[-1]

        with self._get_tracer().start_span(operation_name=op_name, child_of=curr_span) as span:
            span.set_tag('completed', True)

    @staticmethod
    def create_machine():
        ds = DistributedSystem()
        Machine(ds, states=ds.states, transitions=ds.transitions, initial=ds.init_state)
        return ds


def init_tracer(service):
    """Initialize and configure Jaeger’s Python bindings to send traces"""

    log_level = logging.WARNING
    logging.getLogger('').handlers = []
    logging.basicConfig(format='%(message)s', level=log_level)

    config = Config(
        config={
            'sampler': {
                'type': 'const',
                'param': 1,
            },
            'logging': True,
        },
        service_name=service,
    )
    return config.initialize_tracer()


def main():
    ds = DistributedSystem.create_machine()
    ds.get_job()
    ds.get_location()
    ds.publish()
    # Since the tracer does not flush the span immediately to the trace server, we insert a pause to give time the
    # program to dispatch the span. Otherwise, the program may exit before the tracer send the span.
    input("Press Enter to continue...")


if __name__ == "__main__":
    main()
