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
    case $tag in
        v0.0.1)
            git checkout $tag
            python docs/update_avocado_rst.py
            sphinx-build docs/source docs/api/$tag
            rm -rf docs/source
            ;;
        v0.0.2-alpha)
            git checkout $tag
            python docs/avocado/update_avocado_rst.py
            sphinx-build docs/avocado/sphinx_src docs/api/$tag
            rm -rf docs/avocado/sphinx_src
            ;;
        v0.1.0 | *)
            git checkout $tag
            python docs/update_quairkit_rst.py
            sphinx-build docs/sphinx_src docs/api/$tag
            rm -rf docs/sphinx_src
            ;;
        v0.2.0-alpha | *)
            git checkout $tag
            python docs/update_quairkit_rst.py
            sphinx-build docs/sphinx_src docs/api/$tag
            rm -rf docs/sphinx_src
            ;;
        v0.2.0 | *)
            git checkout $tag
            python docs/update_quairkit_rst.py
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
