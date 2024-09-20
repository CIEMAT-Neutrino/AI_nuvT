#include <TFile.h>
#include <TTree.h>
#include <TBranch.h>
#include <iostream>

// Function to convert bytes to a human-readable format
std::string getReadableSize(Long64_t size) {
    const char* units[] = {"B", "KB", "MB", "GB", "TB"};
    int unitIndex = 0;
    double humanReadableSize = size;

    while (humanReadableSize >= 1024 && unitIndex < 4) {
        humanReadableSize /= 1024;
        unitIndex++;
    }

    return std::to_string(humanReadableSize) + " " + units[unitIndex];
}

void print_ttree_with_sizes() {
    // Open the ROOT file
    TFile *file = TFile::Open("/data/vidales/opana_tree_combined_v2108.root");
    if (!file || !file->IsOpen()) {
        std::cerr << "Error opening file!" << std::endl;
        return;
    }

    // Get the TTree
    TTree *tree = (TTree*)file->Get("opanatree/OpAnaTree");  // Replace with the correct tree path
    if (!tree) {
        std::cerr << "Tree not found!" << std::endl;
        file->Close();
        return;
    }

    // Print the TTree structure
    std::cout << "Tree Structure:" << std::endl;
    tree->Print();

    // Iterate over branches and print their sizes
    std::cout << "\nBranch Sizes:" << std::endl;
    TObjArray *branches = tree->GetListOfBranches();
    TIter next(branches);
    TBranch *branch;
    while ((branch = (TBranch*)next())) {
        Long64_t branchSize = branch->GetExpectedSize();  // Size in bytes
        std::cout << "Branch: " << branch->GetName() << " Size: " << getReadableSize(branchSize) << std::endl;
    }

    // Close the file
    file->Close();
}
