#!/bin/bash

# Check and install the Sphinx Material theme and nbsphinx if not already installed
pip show sphinx_immaterial || pip install sphinx_immaterial
pip show nbsphinx || pip install nbsphinx

# Clean up the existing API documentation directories
rm -rf docs/api
rm -rf docs/sphinx_src
rm -rf docs/source

# Retrieve all Git tags and store them in an array
tags=($(git tag))
echo "Tags available: ${tags[*]}"

# Retrieve the name of the current branch
current_branch=$(git rev-parse --abbrev-ref HEAD)

# Initialize the version_info list for Python configuration
version_info_content="html_theme_options['version_info'] = ["

# Populate the version_info list with tag data
for i in "${!tags[@]}"; do
    tag="${tags[$i]}"
    if [ $i -eq 0 ]; then
        version_info_content+="{'version': '$tag', 'title': '$tag', 'aliases': ['$tag']}"
    else
        version_info_content+=", {'version': '$tag', 'title': '$tag', 'aliases': ['$tag']}"
    fi
done

# Close the version_info list
version_info_content+="]"

# Fetch all tags from the repository
git fetch --all --tags

# Loop through all tags and generate Sphinx documentation for each with specific conditions
for tag in "${tags[@]}"; do
    # Skip the "latest" tag
    if [ "$tag" == "latest" ]; then
        continue
    fi

    echo "Switching to tag $tag..."
    git checkout "$tag"
    if [ $? -ne 0 ]; then
        echo "Failed to switch to tag $tag. Skipping..."
        continue
    fi

    # Generate the Sphinx documentation for the tag
    case $tag in
        v0.1.0)
            python docs/update_quairkit_rst.py
            sphinx-build docs/sphinx_src docs/api/$tag
            rm -rf docs/sphinx_src
            ;;
        v0.2.0)
            python docs/update_quairkit_rst.py
            sphinx-build docs/sphinx_src docs/api/$tag
            rm -rf docs/sphinx_src
            ;;
        v0.3.0 | *)
            python docs/update_quairkit_rst.py
            cp -r tutorials docs/sphinx_src/
            sphinx-build docs/sphinx_src docs/api/$tag
            rm -rf docs/sphinx_src
            ;;
        v0.4.0 | *)
            python docs/update_quairkit_rst.py
            cp -r tutorials docs/sphinx_src/
            sphinx-build docs/sphinx_src docs/api/$tag
            rm -rf docs/sphinx_src
            ;;
        v0.4.1 | *)
            python docs/update_quairkit_rst.py
            cp -r tutorials docs/sphinx_src/
            sphinx-build docs/sphinx_src docs/api/$tag
            rm -rf docs/sphinx_src
            ;;
        v0.4.2 | *)
            python docs/update_quairkit_rst.py
            cp -r tutorials docs/sphinx_src/
            sphinx-build docs/sphinx_src docs/api/$tag
            rm -rf docs/sphinx_src
            ;;
    esac
done

# Revert to the current branch
git checkout $current_branch

# Build the final Sphinx documentation for the current branch
python docs/update_quairkit_rst.py
cp -r tutorials docs/sphinx_src/
echo "$version_info_content" >> docs/sphinx_src/conf.py
sphinx-build docs/sphinx_src docs/api/latest

# Notify the user where to find the built documentation
echo "The documentation is built. You can open it by navigating to 'docs/api/index.html' in your browser."
