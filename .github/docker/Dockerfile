FROM quay.io/pypa/manylinux_2_28_x86_64

# Install Go
RUN curl -L https://golang.org/dl/go1.21.0.linux-amd64.tar.gz | tar -xz -C /usr/local && \
    ln -s /usr/local/go/bin/go /usr/local/bin/go

# Install auditwheel (using a consistent Python)
RUN /opt/python/cp310-cp310/bin/pip install auditwheel

WORKDIR /workspace

# Copy the build script
COPY build.sh /build.sh
RUN chmod +x /build.sh

ENTRYPOINT ["/build.sh"]