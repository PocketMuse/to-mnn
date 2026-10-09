FROM template_generator

WORKDIR /workspace
COPY --from=converter /usr/local/bin/sd15-convert /usr/local/bin/sd15-convert
COPY --from=converter /usr/local/bin/sd15-api-test /usr/local/bin/sd15-api-test
COPY tests /workspace/tests
ENV SD15_CONVERTER=/usr/local/bin/sd15-convert
ENTRYPOINT ["python", "-m", "unittest", "discover", "-s", "tests", "-v"]
