Distributed Tracing
===================


Prerequisites
--------------

Install the Jaeger platform using a Docker image. Test your deployment using the Jaeger UI: http://localhost:16686

.. code-block:: console

    $ docker run -d -p5775:5775/udp -p6831:6831/udp -p6832:6832/udp -p5778:5778 -p16686:16686 -p14268:14268 -p9411:9411 jaegertracing/all-in-one:0.8.0


Install Jaeger's implementation of OpenTracing library for Python to send traces to Jaeger's backend.
Alternatives Jaeger include Datadog or Lightstep's libraries.

.. code-block:: console

    $  pip install jaeger-client


Install an object-oriented state machine implementation to simulate the execution of a distributed application.

.. code-block:: console

    $  pip install transitions


Run the Example
---------------

.. code-block:: console

    $ python enrich_jobs_app.py