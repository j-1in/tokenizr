#include "bpe.h"
#include "absl/log/globals.h"
#include "absl/log/initialize.h"

int main(int argc, char** argv) {
    absl::InitializeLog();
    absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);
    return bpe::run_cli(argc, argv);
}
